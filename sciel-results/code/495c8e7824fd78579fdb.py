import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    
    # Precompute squared norms for the grid points
    # This is useful if we were doing distance calculations more directly, 
    # but for the determinant check, we compute relative norms on the fly.
    
    def check_validity(arr: np.ndarray, p: np.ndarray) -> bool:
        """
        Check if point p can be added to arr without creating 5 co-spherical points.
        arr: (m, 3) array of existing points.
        p: (3,) array of candidate point.
        Returns True if valid.
        """
        m = len(arr)
        if m < 4:
            return True
        
        # Lifted vectors relative to p:
        # w_i = (arr[i] - p, |arr[i]|^2 - |p|^2)
        # We need to check if any 4 of these vectors are linearly dependent.
        
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        
        # Generate all combinations of 4 indices from range(m)
        # For m=50, C(50,4) = 230300. This is manageable.
        # For m=100, C(100,4) = 3921225. This might be slow if done too often.
        
        indices = np.arange(m)
        # Use itertools to generate combinations, then convert to numpy array for indexing
        # This avoids storing a massive list of tuples in Python memory if we can vectorize the indexing
        
        # Optimization: If m is large, this step is the bottleneck.
        # Let's try to generate the indices efficiently.
        
        # For very large m, we might need to sample combinations, but the problem requires ALL combinations.
        # So we must check all.
        
        # Convert combinations to a numpy array for fancy indexing
        # This creates an array of shape (C(m,4), 4)
        try:
            combos = np.array(list(itertools.combinations(indices, 4)), dtype=np.int64)
        except MemoryError:
            # Fallback for extremely large m (unlikely in this grid size context)
            return False
            
        if len(combos) == 0:
            return True
            
        # Extract sub-matrices
        # w[combos] will have shape (C(m,4), 4, 4)
        sub_matrices = w[combos]
        
        # Compute determinants
        # np.linalg.det on a batch of matrices
        dets = np.linalg.det(sub_matrices)
        
        # Check if any determinant is too small
        if np.any(np.abs(dets) < 0.5):
            return False
            
        return True

    def greedy_farthest_topk(seed: int, top_k: int = 20) -> np.ndarray:
        """
        Greedy strategy using Farthest-Point Sampling.
        At each step, pick the top_k furthest points from the current set
        and try to add the first valid one.
        """
        rng = np.random.RandomState(seed)
        
        # Start with a random point
        start_idx = rng.randint(num_points_total)
        arr = grid[start_idx:start_idx+1].copy()
        
        # Set of available points (indices)
        available = set(range(num_points_total))
        available.remove(start_idx)
        
        while available:
            # Compute min distance from each available point to any point in arr
            avail_indices = list(available)
            if not avail_indices:
                break
                
            avail_grid = grid[avail_indices]
            
            # Distance squared
            # diff[i, j, k] = avail_grid[i, k] - arr[j, k]
            diff = avail_grid[:, np.newaxis, :] - arr[np.newaxis, :, :]
            dist_sq = (diff ** 2).sum(axis=2) # Shape: (num_avail, m)
            min_dist_sq = dist_sq.min(axis=1) # Shape: (num_avail,)
            
            # Get indices of top_k largest min_dist_sq
            # We want the furthest ones.
            # argsort descending
            sorted_indices = np.argsort(-min_dist_sq)
            
            added = False
            # Try top_k candidates
            for idx_in_avail in sorted_indices[:top_k]:
                orig_idx = avail_indices[idx_in_avail]
                p = grid[orig_idx]
                
                if check_validity(arr, p):
                    arr = np.vstack([arr, p])
                    available.remove(orig_idx)
                    added = True
                    break
            
            if not added:
                # If top_k didn't work, we can try a few random ones from the rest
                # or just stop. Stopping is safer for time.
                # Let's try 5 random ones to break potential local minima
                if len(available) > top_k:
                    remaining_indices = sorted_indices[top_k:]
                    # Pick 5 random from remaining
                    try_indices = rng.choice(len(remaining_indices), size=min(5, len(remaining_indices)), replace=False)
                    for idx_in_rem in try_indices:
                        orig_idx = avail_indices[remaining_indices[idx_in_rem]]
                        p = grid[orig_idx]
                        if check_validity(arr, p):
                            arr = np.vstack([arr, p])
                            available.remove(orig_idx)
                            added = True
                            break
                            
                if not added:
                    break
                    
        return arr

    start_time = time.time()
    time_limit = 600.0
    safety_margin = 10.0
    
    best = np.empty((0, 3), dtype=np.int64)
    
    # Phase 1: Run several restarts with different top_k values
    # Smaller top_k is faster but might get stuck. Larger top_k is slower but better.
    # We'll alternate or try a range.
    
    seeds = range(100)
    top_k_values = [10, 20, 50]
    
    for seed in seeds:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        # Cycle through top_k values
        k = top_k_values[seed % len(top_k_values)]
        
        res = greedy_farthest_topk(seed, k)
        
        if len(res) > len(best):
            best = res
            # Optional: Print progress to monitor (disabled for final output)
            # print(f"Seed {seed}, K {k}: Found {len(res)} points")

    return best.astype(np.int64)
