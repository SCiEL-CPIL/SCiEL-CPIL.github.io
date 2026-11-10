# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    """
    Returns all integer vectors in Z^11 with squared norm 4.
    These are:
    1. ±2e_i (22 points)
    2. ±e_i ± e_j ± e_k ± e_l (5280 points)
    Total k = 5302.
    Min squared distance is 4. Max squared norm is 4.
    Condition 4 >= 4 is satisfied.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    d = 11
    points_list = []
    
    # Type 1: ±2e_i
    for i in range(d):
        v1 = [0] * d
        v1[i] = 2
        points_list.append(v1)
        v2 = [0] * d
        v2[i] = -2
        points_list.append(v2)
        
    if time.time() > deadline - 10:
        return np.array(points_list, dtype=np.int64)

    # Type 2: ±e_i ± e_j ± e_k ± e_l
    # Choose 4 distinct indices
    indices = range(d)
    for combo in itertools.combinations(indices, 4):
        i, j, k, l = combo
        # 16 sign combinations
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        v = [0] * d
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points_list.append(v)
                        
    points = np.array(points_list, dtype=np.int64)
    
    # Quick verification
    if points.shape[0] < 2:
        return points
        
    norms2 = np.sum(points**2, axis=1)
    max_norm2 = int(np.max(norms2))
    
    k = points.shape[0]
    min_dist2 = float('inf')
    
    # Spot check a few pairs to ensure no bugs, full check is expensive
    # But 5302 points -> ~14M pairs.
    # 14M iterations in Python might take ~10-30 seconds.
    # We have 600s budget.
    # Let's do a full check to be safe, but break early if invalid.
    
    if time.time() < deadline - 60:
        for i in range(k):
            for j in range(i + 1, k):
                diff = points[i] - points[j]
                d2 = int(np.dot(diff, diff))
                if d2 < min_dist2:
                    min_dist2 = d2
                    if min_dist2 < max_norm2:
                        # Invalid, fallback to smaller set
                        return np.array([], dtype=np.int64).reshape(0, d)
    else:
        # If time is tight, skip verification and trust the math
        pass
        
    if np.any(np.all(points == 0, axis=1)):
        return points[:0]
        
    if min_dist2 < max_norm2:
        return points[:0]
        
    points = np.around(points).astype(np.int64)
    return points

# EVOLVE-BLOCK-END
