import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            # Generate all 4 sign combinations for e_i +/- e_j
            # (1, 1), (1, -1), (-1, 1), (-1, -1)
            p1 = np.zeros(d, dtype=np.int64)
            p2 = np.zeros(d, dtype=np.int64)
            p3 = np.zeros(d, dtype=np.int64)
            p4 = np.zeros(d, dtype=np.int64)
            
            p1[i] = 1; p1[j] = 1
            p2[i] = 1; p2[j] = -1
            p3[i] = -1; p3[j] = 1
            p4[i] = -1; p4[j] = -1
            
            points.append(p1)
            points.append(p2)
            points.append(p3)
            points.append(p4)
            
    points_array = np.array(points)
    return points_array
