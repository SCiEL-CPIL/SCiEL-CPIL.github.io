# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs 11-dimensional integer points: all e_i + e_j and e_i - e_j for i < j.
    All have norm² = 2, and minimum pairwise squared distance is 2, satisfying
    min_dist² >= max_norm².
    
    Returns:
        points: np.ndarray of shape (110, 11) with int64 entries
    """
    d = 11
    points_list = []
    
    for i in range(d):
        for j in range(i + 1, d):
            # e_i + e_j
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1
            v1[j] = 1
            points_list.append(v1)
            # e_i - e_j
            v2 = np.zeros(d, dtype=np.int64)
            v2[i] = 1
            v2[j] = -1
            points_list.append(v2)
    
    points = np.array(points_list, dtype=np.int64)
    return points


# EVOLVE-BLOCK-END
