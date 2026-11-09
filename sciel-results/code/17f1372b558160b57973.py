import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0
    
    # 1. Try to find a large set of points with squared norm 4
    # Candidates: vectors with exactly 4 non-zero coordinates, each +/-1
    # Count: C(11, 4) * 2^4 = 330 * 16 = 5280
    
    d = 11
    candidates = []
    indices = range(d)
    
    # Generate combinations of 4 indices
    # Using itertools for clarity, or manual loops
    import itertools
    
    for combo in itertools.combinations(indices, 4):
        i, j, k, l = combo
        for si in [-1, 1]:
            for sj in [-1, 1]:
                for sk in [-1, 1]:
                    for sl in [-1, 1]:
                        p = [0] * d
                        p[i] = si
                        p[j] = sj
                        p[k] = sk
                        p[l] = sl
                        candidates.append(p)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    if time.time() > deadline - 60:
        return _get_norm2_points()

    # Compute dot products
    # dist_sq = 4 + 4 - 2 dot = 8 - 2 dot
    # We need dist_sq >= 4 => 8 - 2 dot >= 4 => 2 dot <= 4 => dot <= 2
    # Conflict if dot > 2 (i.e., dot >= 3)
    
    try:
        dot_prods = candidates @ candidates.T
        # Conflict if dot > 2
        conflict_mask = dot_prods > 2
        
        # Precompute neighbors for each node to speed up greedy blocking
        # neighbors[i] is a list of indices j such that conflict_mask[i, j] is True
        neighbors = [np.where(conflict_mask[i])[0] for i in range(n_cand)]
        
        best_size = 0
        best_set = []
        
        # Run multiple random greedy restarts
        # The greedy algorithm:
        # 1. Shuffle indices
        # 2. Iterate through shuffled indices
        # 3. If index is available, select it, mark it and its neighbors as unavailable
        
        num_restarts = 100
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            order = np.random.permutation(n_cand)
            available = np.ones(n_cand, dtype=bool)
            selected = []
            
            for idx in order:
                if available[idx]:
                    selected.append(idx)
                    available[idx] = False
                    # Block neighbors
                    # Using the precomputed list
                    if len(neighbors[idx]) > 0:
                        available[neighbors[idx]] = False
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected
                
    except Exception as e:
        # Fallback
        return _get_norm2_points()

    if best_size > 220:
        points = candidates[best_set]
        # Verify
        norms_sq = np.sum(points**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        min_dist_sq = float('inf')
        n_points = len(points)
        # Optimization: If n_points is large, this loop is slow.
        # But we only do this if we found a set > 220.
        # Let's use vectorized check for min dist if possible?
        # Or just sample? No, must be exact.
        # If best_size is e.g. 500, 500*499/2 ~ 125k pairs. Fast.
        # If best_size is 2000, 2M pairs. Still okay in 600s? 
        # NumPy sum of diff**2 is fast.
        
        # To be safe, if best_size is very large, we might want to verify more efficiently.
        # But let's assume the greedy result is valid and check a few pairs first?
        # No, the contract says "all checks use these integers with exact arithmetic".
        # We must ensure validity.
        
        # Efficient verification:
        # We can compute all pairwise distances?
        # (n, 11) x (11, n) -> (n, n) dot product.
        # dist^2[i, j] = norm[i] + norm[j] - 2 dot[i, j]
        # We need min(dist^2) >= max(norm^2).
        
        norms = np.sum(points**2, axis=1)
        max_norm = np.max(norms)
        
        dots = points @ points.T
        dists_sq = norms[:, None] + norms[None, :] - 2 * dots
        
        # Set diagonal to inf to ignore self-distances
        np.fill_diagonal(dists_sq, float('inf'))
        
        min_dist = np.min(dists_sq)
        
        if min_dist >= max_norm:
            return points

    # Fallback to R^2 = 2
    return _get_norm2_points()

def _get_norm2_points():
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = [0] * d
                    p[i] = si
                    p[j] = sj
                    points.append(p)
    return np.array(points, dtype=np.int64)
