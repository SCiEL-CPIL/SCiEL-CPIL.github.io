# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Returns the set of all 220 weight-2 integer vectors in 11 dimensions.
    These vectors have norm^2 = 2.
    The minimum squared distance between any pair is 2 (if they share an index with same sign)
    or 4 (if they share an index with opposite sign or are disjoint).
    Since min_dist_sq (2) >= max_norm_sq (2), the configuration is valid.
    """
    d = 11
    points = []
    
    # Generate all weight-2 vectors: (i, j) with signs (si, sj)
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points.append(p)
    
    points = np.array(points, dtype=np.int64)
    
    # Basic checks
    k = len(points)
    if k < 2:
        return np.zeros((2, d), dtype=np.int64) # Fallback, though unlikely
    
    # Verify validity to be safe
    max_norm_sq = 0
    for p in points:
        ns = int(np.dot(p, p))
        if ns > max_norm_sq:
            max_norm_sq = ns
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            ds = int(np.dot(diff, diff))
            if ds < min_dist_sq:
                min_dist_sq = ds
    
    # The condition is min_dist_sq >= max_norm_sq
    # For weight-2 vectors, min_dist_sq = 2, max_norm_sq = 2.
    # 2 >= 2 is True.
    
    return points

# EVOLVE-BLOCK-END
