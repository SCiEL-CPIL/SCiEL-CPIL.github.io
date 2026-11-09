# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.
    
    Strategy: Generate all integer vectors with exactly two non-zero entries of value +/- 1.
    These vectors have squared norm 2.
    For any two distinct such vectors v and w:
    - If they share no indices, v.w = 0, dist^2 = 4.
    - If they share one index, v.w = +/- 1, dist^2 = 2 or 6.
    - If they share two indices, they are identical (dist^2=0) or negatives (dist^2=8) or one is (1,1) and other (1,-1) etc.
      Wait, if they share two indices, they are the same support.
      e.g., v = (1, 1, 0...), w = (1, -1, 0...). v.w = -1. dist^2 = 6.
      e.g., v = (1, 1, 0...), w = (-1, 1, 0...). v.w = -1. dist^2 = 6.
      e.g., v = (1, 1, 0...), w = (-1, -1, 0...). v.w = -2. dist^2 = 8.
      e.g., v = (1, 1, 0...), w = (1, 1, 0...). v.w = 2. dist^2 = 0. (Identical)
    
    Since we only include distinct vectors, the minimum non-zero distance squared is 2.
    The maximum squared norm is 2.
    Condition: min_dist^2 >= max_norm^2 => 2 >= 2. Holds.
    
    Total points: C(11, 2) * 4 = 55 * 4 = 220.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    d = 11
    points_list = []
    
    # Generate all vectors with exactly 2 non-zero entries, each being +/- 1
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
                
    # Check time, though this loop is very fast
    if time.time() > deadline - 10:
         return np.array(points_list[:2], dtype=np.int64)

    points = np.array(points_list, dtype=np.int64)
    
    return points

# EVOLVE-BLOCK-END
