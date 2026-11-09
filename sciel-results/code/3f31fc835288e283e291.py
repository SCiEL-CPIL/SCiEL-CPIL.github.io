# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product


def kissing_number11() -> np.ndarray:
    """
    Constructs the set of 220 eleven-dimensional integer points with exactly
    two non-zero entries (each +/-1).
    
    Validity Check:
    - Max squared norm N = 1^2 + 1^2 = 2.
    - Min pairwise squared distance:
      - If supports are disjoint: dist^2 = 2 + 2 = 4 >= 2.
      - If supports share 1 index: dist^2 = (1-1)^2 + 1^2 + 1^2 = 2 >= 2 (if signs match on shared index)
        or dist^2 = (1-(-1))^2 + 1^2 + 1^2 = 6 >= 2 (if signs differ).
      - If supports are identical: dist^2 = 0 (if same) or 4 (if one sign flipped) or 8 (if both flipped).
        Since we include all sign variations, we include distinct points.
        Pairs with same support and different signs have dist^2 >= 4.
        Pairs with same support and same signs are identical (not distinct).
        So min distance between distinct points is 2.
    - Condition: min_dist_sq >= max_norm_sq => 2 >= 2. Holds.
    - k = 220.
    """
    d = 11
    points_list = []
    for i, j in combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
    
    best_points = np.array(points_list, dtype=np.int64)
    best_points = np.around(best_points).astype(np.int64)
    return best_points


# EVOLVE-BLOCK-END
