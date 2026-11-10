import numpy as np
import time

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    for i in range(d):
        for j in range(i+1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    points.append(v)
    
    points = np.array(points, dtype=np.int64)
    return points
