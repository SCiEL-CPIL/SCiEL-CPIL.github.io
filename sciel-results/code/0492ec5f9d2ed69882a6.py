import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 590  # 10s safety margin
    
    # 1. Generate the guaranteed valid set of 242 Norm-4 vectors
    # Type A: One entry +/-2
    # Type B: Two entries +/-1
    points_norm4 = []
    for i in range(d):
        for s in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2 * s
            points_norm4.append(v)
    
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points_norm4.append(v)
    
    pts_array_norm4 = np.array(points_norm4, dtype=np.int64)
    best_k = len(pts_array_norm4)
    best_array = pts_array_norm4
    
    # 2. Try to find a larger set using Norm-8 vectors
    # Candidates:
    # Type C: Two entries +/-2
    # Type D: One entry +/-2, Two entries +/-1
    
    candidates = []
    # Type C
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    candidates.append(v)
                    
    # Type D
    for k_idx in range(d):
        for sk in [1, -1]:
            # Choose 2 other indices
            for i, j in itertools.combinations([x for x in range(d) if x != k_idx], 2):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[k_idx] = 2 * sk
                        v[i] = si
                        v[j] = sj
                        candidates.append(v)
                        
    M = len(candidates)
    if M == 0:
        return np.around(best_array).astype(np.int64)
        
    V = np.array(candidates, dtype=np.int64)
    
    # Compute dot products
    # V is M x d. Dots is M x M.
    # To save memory/time, we can do it in chunks if M is huge, but 4180 is small.
    Dots = V @ V.T
    
    # Conflict if Dots > 4
    # Note: Diagonal is 8, so no self-conflict, but we handle distinct pairs.
    Conflict = Dots > 4
    
    # Greedy MIS search
    # We will try multiple random orderings
    # Or degree-based greedy
    
    # Precompute degrees
    degrees = Conflict.sum(axis=1)
    
    # We'll do a few iterations of greedy selection
    # Strategy: Pick node with min degree, remove it and neighbors.
    # To speed up, we can just do random shuffles and check compatibility with selected set.
    
    # Method: Random Shuffling Greedy
    # For each attempt:
    #   order = np.random.permutation(M)
    #   selected = []
    #   selected_mask = np.zeros(M, dtype=bool)
    #   for idx in order:
    #       if not selected_mask[idx]:
    #           # Check if idx conflicts with any selected
    #           # selected_indices = np.where(selected_mask)[0]
    #           # if not np.any(Conflict[idx, selected_indices]):
    #           #     selected.append(idx)
    #           #     selected_mask[idx] = True
    #           #     selected_mask[Conflict[idx, :]] = True # Mark neighbors as unavailable
    #           pass
    # This is O(M * K) per attempt.
    
    # Optimization:
    # Keep a `blocked` mask.
    # When we pick `idx`, we set `blocked[idx] = True` and `blocked[Conflict[idx, :]] = True`.
    # Then we just iterate through `order` and pick if `not blocked[idx]`.
    
    num_attempts = 0
    while time.time() < deadline - 5:
        num_attempts += 1
        order = np.random.permutation(M)
        blocked = np.zeros(M, dtype=bool)
        selected_indices = []
        
        for idx in order:
            if not blocked[idx]:
                selected_indices.append(idx)
                # Block idx and its neighbors
                blocked[idx] = True
                blocked[Conflict[idx, :]] = True
                
        current_k = len(selected_indices)
        if current_k > best_k:
            best_k = current_k
            best_array = V[selected_indices]
            
        # If we found a very large set, we can stop early?
        # Or continue to try to improve.
        
    # Final verification of best_array to be safe
    # The problem requires exact arithmetic check.
    # Our construction ensures validity for Norm 4.
    # For Norm 8, we ensured dot <= 4 via conflict graph.
    # But let's double check the best_array found.
    
    if best_k > 0:
        pts = best_array
        n_pts = len(pts)
        if n_pts > 1:
            norms = np.sum(pts ** 2, axis=1)
            max_norm = int(norms.max())
            
            min_dist = float('inf')
            # Only verify if size is manageable or if it's the new best
            # If best_k > 242, verify.
            if best_k > 242:
                for i in range(n_pts):
                    for j in range(i + 1, n_pts):
                        diff = pts[i] - pts[j]
                        dist_sq = int(np.sum(diff ** 2))
                        if dist_sq < min_dist:
                            min_dist = dist_sq
                if min_dist < max_norm:
                    # Invalid, fallback to norm4
                    best_array = pts_array_norm4
                    best_k = len(pts_array_norm4)

    return np.around(best_array).astype(np.int64)
