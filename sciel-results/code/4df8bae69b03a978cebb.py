# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points satisfying:
    - min pairwise squared distance >= max squared norm
    
    Structural approach: Use ALL norm-2 vectors (exactly two ±1 entries).
    For any two distinct norm-2 vectors v, w in Z^11, v·w ∈ {-2,-1,0,1},
    so ||v-w||² = 4 - 2(v·w) ∈ {2, 4, 6, 8} >= 2 = max norm².
    Thus all C(11,2)*4 = 220 such vectors form a valid kissing configuration.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    d = 11
    points = []
    
    # All norm-2 vectors: exactly two non-zero entries, each ±1
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                points.append(v)
    
    result = np.array(points, dtype=np.int64)
    result = np.around(result).astype(np.int64)
    return result


# EVOLVE-BLOCK-END
