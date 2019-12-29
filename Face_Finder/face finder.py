import os
import cv2
import face_recognition
import numpy as np
from time import sleep
# run python face Identification.py

def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = face_recognition.load_image_file("faces/" + f)
                encoding = face_recognition.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded




def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_location = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_location)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
        #true,false
        print(matches)

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        # array value
        print(face_distances)

        best_match_index = np.argmin(face_distances)
        # best_match_index = np.array(face_distances)
        # best_match_index=best_match_index.astype(int)

        # # best index image
        print(best_match_index)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
    return face_names


print(classify_face("tt.JPG"))


