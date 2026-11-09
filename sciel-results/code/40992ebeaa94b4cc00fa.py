import numpy as np

N = 50

def construct():
        # Indices of a 3-AP free Sidon set within 0..49
        # Based on base-3 representation with digits 0,1
        indices = [0, 1, 3, 4, 9, 10, 12, 13, 27, 28, 30, 31, 36, 37, 39, 40]
        h = [0.0] * N
        for i in indices:
            h[i] = 1.0 / len(indices)
        return h
