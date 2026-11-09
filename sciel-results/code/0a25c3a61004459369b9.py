# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer vectors with exactly two
    non-zero entries of value +/-1. All have squared norm 2.
    
    For any two distinct such vectors:
    - Share 0 indices: dot=0, dist^2=4
    - Share 1 index: dot=+/-1, dist^2=2 or 6
    - Share 2 indices: same support, distinct sign patterns, dot in {-2,-1,1,2}
      but distinct so dot in {-2,-1,1}, dist^2 in {4,6,8}
    
    Min pairwise dist^2 = 2, max norm^2 = 2. Condition 2 >= 2 holds.
    Total: C(11,2)*4 = 220 points.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    d = 11
    points_list = []
    
    # Generate all vectors with exactly 2 non-zero entries, each being +/- 1
    for i, j in combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
    
    if time.time() > deadline:
        # Fallback: return a small valid set
        return np.array(points_list[:2], dtype=np.int64)

    points = np.array(points_list, dtype=np.int64)
    return points

# EVOLVE-BLOCK-END
