import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s limit

    d = 11
    points_list = []

    # 1. All 220 vectors of type (+/-2, +/-2, 0, ...) in Z^11
    # These are 2 * (roots of D_11)
    for i, j in combinations(range(11), 2):
        if time.time() > deadline:
            break
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)

    # 2. All 128 vectors of type (+/-1, ..., +/-1, 0, 0, 0) in first 8 dims
    # These are 2 * (Type 2 roots of E_8) -- actually D_8 type
    for signs in product([-1, 1], repeat=8):
        if time.time() > deadline:
            break
        if sum(1 for s in signs if s == -1) % 2 == 0:
            p = np.zeros(d, dtype=np.int64)
            for idx, val in enumerate(signs):
                p[idx] = val
            points_list.append(p)

    points = np.array(points_list, dtype=np.int64)
    current_norm_sq = 8
    max_norm = 8

    # Generate all candidate vectors with norm squared 8
    candidates = []
    
    # Type 1: Four 1s
    for idxs in combinations(range(11), 4):
        for signs in product([-1, 1], repeat=4):
            p = np.zeros(d, dtype=np.int64)
            for idx, val in zip(idxs, signs):
                p[idx] = val
            candidates.append(p)
            
    # Type 2: Two 2s
    for idxs in combinations(range(11), 2):
        for signs in product([-1, 1], repeat=2):
            p = np.zeros(d, dtype=np.int64)
            for idx, val in zip(idxs, signs):
                p[idx] = 2 * val
            candidates.append(p)

    # Type 3: One 2, four 1s
    for idx2 in range(11):
        for idxs1 in combinations([i for i in range(11) if i != idx2], 4):
            for s2 in [-1, 1]:
                for signs in product([-1, 1], repeat=4):
                    p = np.zeros(d, dtype=np.int64)
                    p[idx2] = 2 * s2
                    for idx, val in zip(idxs1, signs):
                        p[idx] = val
                    candidates.append(p)

    # Type 4: Eight 1s
    for idxs in combinations(range(11), 8):
        for signs in product([-1, 1], repeat=8):
            p = np.zeros(d, dtype=np.int64)
            for idx, val in zip(idxs, signs):
                p[idx] = val
            candidates.append(p)
            
    candidates = np.array(candidates, dtype=np.int64)
    
    # Remove candidates already in points
    # Efficient check:
    # points shape (N, 11), candidates shape (M, 11)
    # We can use a set of tuples for points
    point_set = set(map(tuple, points))
    unique_candidates = []
    for c in candidates:
        if tuple(c) not in point_set:
            unique_candidates.append(c)
            
    if unique_candidates:
        unique_candidates = np.array(unique_candidates, dtype=np.int64)
    else:
        unique_candidates = np.empty((0, d), dtype=np.int64)

    # Greedily add candidates
    # We maintain `points` as the current valid set.
    # For each candidate, check if min dist to `points` >= 8.
    # To speed up, we process in chunks.
    
    added_points = []
    chunk_size = 500
    
    for i in range(0, len(unique_candidates), chunk_size):
        if time.time() > deadline:
            break
        chunk = unique_candidates[i:i+chunk_size]
        
        # Compute squared distances between chunk and points
        # chunk: (C, 11), points: (N, 11)
        # diff: (C, N, 11) -> might be large.
        # C=500, N~350. 500*350*11 = 1.9M floats. OK.
        
        # Broadcasting
        # chunk[:, None, :] - points[None, :, :]
        diff = chunk[:, None, :] - points[None, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        min_dist_sq = np.min(dist_sq, axis=1)
        
        # Valid if min_dist_sq >= 8
        valid_mask = min_dist_sq >= 8
        
        # Add valid ones to added_points and update points
        if np.any(valid_mask):
            valid_chunk = chunk[valid_mask]
            added_points.extend([tuple(p) for p in valid_chunk])
            points = np.vstack([points, valid_chunk])
            
    # Final verification
    final_norms = np.sum(points**2, axis=1)
    final_max_norm = np.max(final_norms)
    final_n = len(points)

    final_min_dist_sq = float('inf')
    for i in range(final_n):
        if time.time() > deadline:
            break
        for j in range(i + 1, final_n):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < final_min_dist_sq:
                final_min_dist_sq = dist_sq
                if final_min_dist_sq < final_max_norm:
                    break
        if final_min_dist_sq < final_max_norm:
            break

    if final_min_dist_sq < final_max_norm:
        # Fallback to original if broken (should not happen with greedy on norm 8)
        points = np.array(points_list, dtype=np.int64)

    result = np.around(points).astype(np.int64)
    return result
