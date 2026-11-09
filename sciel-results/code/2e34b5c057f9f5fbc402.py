import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            # e_i + e_j
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            v[j] = 1
            points.append(v)
            
            # e_i - e_j
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            v[j] = -1
            points.append(v)
            
            # -e_i + e_j
            v = np.zeros(d, dtype=np.int64)
            v[i] = -1
            v[j] = 1
            points.append(v)
            
            # -e_i - e_j
            v = np.zeros(d, dtype=np.int64)
            v[i] = -1
            v[j] = -1
            points.append(v)
            
    return np.array(points)
