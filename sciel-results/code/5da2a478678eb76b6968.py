import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses the set of all 220 vectors with squared norm 2 (vectors with two non-zero
    entries, each +/-1).
    
    For this set, the maximum squared norm R is 2.
    The minimum squared distance between any distinct pair is 2.
    Thus, min_dist^2 >= R holds (2 >= 2).
    """
    d = 11
    points = []
    
    for i in range(d):
        for j in range(i + 1, d):
            # Generate 4 vectors for each pair of indices (i, j)
            # 1. e_i + e_j
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1
            v1[j] = 1
            points.append(v1)
            
            # 2. e_i - e_j
            v2 = np.zeros(d, dtype=np.int64)
            v2[i] = 1
            v2[j] = -1
            points.append(v2)
            
            # 3. -e_i + e_j
            v3 = np.zeros(d, dtype=np.int64)
            v3[i] = -1
            v3[j] = 1
            points.append(v3)
            
            # 4. -e_i - e_j
            v4 = np.zeros(d, dtype=np.int64)
            v4[i] = -1
            v4[j] = -1
            points.append(v4)
            
    points_arr = np.array(points)
    return points_arr
