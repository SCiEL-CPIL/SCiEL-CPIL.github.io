import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    based on the 2E8 ⊕ 2D3 lattice minimal vectors.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep safety margin
    
    d = 11
    points = []
    
    # Generate 2E8 minimal vectors (scaled by 2)
    # Type 1: (±2, ±2, 0, ..., 0)
    # Choose 2 positions for the 2s
    for pos in itertools.combinations(range(8), 2):
        p1, p2 = pos
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(8, dtype=np.int64)
                v[p1] = 2 * s1
                v[p2] = 2 * s2
                # Embed in 11D: (v, 0, 0, 0)
                point = np.zeros(11, dtype=np.int64)
                point[:8] = v
                points.append(point)
                
    # Type 2: (±1, ±1, ..., ±1) with even number of minus signs
    # There are 2^8 = 256 combinations. Half have even minus signs.
    for signs in itertools.product([1, -1], repeat=8):
        # Count minus signs
        num_minus = sum(1 for s in signs if s == -1)
        if num_minus % 2 == 0:
            v = np.array(signs, dtype=np.int64)
            point = np.zeros(11, dtype=np.int64)
            point[:8] = v
            points.append(point)
            
    # Generate 2D3 minimal vectors (scaled by 2)
    # D3 minimal vectors are permutations of (±1, ±1, 0)
    # Scaled by 2: (±2, ±2, 0)
    for pos in itertools.combinations(range(3), 2):
        p1, p2 = pos
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                w = np.zeros(3, dtype=np.int64)
                w[p1] = 2 * s1
                w[p2] = 2 * s2
                # Embed in 11D: (0, 0, 0, 0, 0, 0, 0, 0, w)
                point = np.zeros(11, dtype=np.int64)
                point[8:] = w
                points.append(point)
                
    if len(points) == 0:
        # Fallback to seed
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        
    arr = np.array(points, dtype=np.int64)
    
    # Optional: Verification step (fast for 252 points)
    # k = 252, pairs ~ 31k. Very fast.
    # Check max norm
    norms_sq = np.sum(arr * arr, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Check min dist
    min_dist_sq = np.inf
    k = len(arr)
    for i in range(k):
        for j in range(i + 1, k):
            diff = arr[i] - arr[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if min_dist_sq < max_norm_sq:
                # Invalid, but we expect it to be valid
                break
        if min_dist_sq < max_norm_sq:
            break
            
    if min_dist_sq < max_norm_sq:
        # Fallback if construction failed (should not happen)
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)

    return arr
