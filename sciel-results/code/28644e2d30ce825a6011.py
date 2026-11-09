# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs 220 integer points in 11 dimensions using all 2-support vectors
    (exactly two non-zero entries of ±1). All pairwise squared distances are
    >= 2 = max squared norm, satisfying the constraint.
    
    Returns:
        points: np.ndarray of shape (220, 11) with int64 entries
    """
    d = 11
    points = []
    
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                points.append(v)
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
