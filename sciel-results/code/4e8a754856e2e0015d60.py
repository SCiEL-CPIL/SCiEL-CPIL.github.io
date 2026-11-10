import numpy as np
import itertools
import time
import sys

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    # Generate all grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    
    # Precompute squared norms for grid points
    grid_sq_norms = (grid ** 2).sum(axis=1)
    
    def is_valid_addition(arr_indices, p_idx):
        """
        Checks if adding grid[p_idx] to the set defined by arr_indices violates the constraint.
        arr_indices: array of indices of current points.
        p_idx: index of candidate point.
        """
        m = len(arr_indices)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = grid_sq_norms[p_idx]
        
        # Points in current set
        pts = grid[arr_indices]
        pts_sq = grid_sq_norms[arr_indices]
        
        # Build the matrix W for all points in arr
        # Row: [q - p, |q|^2 - |p|^2]
        # This is (m, 4)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = pts - p
        w[:, 3] = pts_sq - p_sq
        
        # We need to check if any 4 rows have det < 0.5
        # If m is small (< 30), do it directly.
        if m < 30:
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
            return True
        
        # For larger m, we can try to find violations faster.
        # Observation: If a violation exists, it might be found early in a random permutation of indices?
        # But we need determinism and exactness.
        
        # Let's process combinations in batches.
        # To avoid generating a massive list, we iterate.
        # However, iterating C(m, 4) in Python is slow.
        
        # Optimization: 
        # If we sort the indices, we can't easily batch.
        # Let's just do the loop but break early.
        # And use the batched det function.
        
        batch = []
        batch_limit = 1000
        for combo in itertools.combinations(range(m), 4):
            batch.append(combo)
            if len(batch) == batch_limit:
                indices = np.array(batch, dtype=np.int32)
                sub_mats = w[indices]
                dets = np.linalg.det(sub_mats)
                if np.any(np.abs(dets) < 0.5):
                    return False
                batch = []
        if batch:
            indices = np.array(batch, dtype=np.int32)
            sub_mats = w[indices]
            dets = np.linalg.det(sub_mats)
            if np.any(np.abs(dets) < 0.5):
                return False
                
        return True

    def solve():
        start_time = time.time()
        deadline = start_time + 580 # Safety margin
        
        best_points = np.empty((0, 3), dtype=np.int64)
        best_size = 0
        
        np.random.seed(42)
        
        # Phase 1: Initial Greedy Construction
        # We do a few greedy passes to find a good starting point
        for _ in range(5):
            order = np.random.permutation(num_points)
            current_indices = []
            for p_idx in order:
                if is_valid_addition(np.array(current_indices), p_idx):
                    current_indices.append(p_idx)
            
            if len(current_indices) > best_size:
                best_size = len(current_indices)
                best_points = grid[np.array(current_indices)]
        
        # Start Simulated Annealing from the best found so far
        current_indices = list(np.where(np.isin(np.arange(num_points), np.array([np.where((grid == p).all(axis=1))[0][0] for p in best_points])))[0])
        # Actually, simpler: just use the indices from the last greedy run that achieved best_size
        # Let's reconstruct indices for best_points
        current_indices = []
        for p in best_points:
            idx = np.where((grid == p).all(axis=1))[0][0]
            current_indices.append(idx)
        
        current_set = set(current_indices)
        current_size = len(current_indices)
        
        # Simulated Annealing Parameters
        T_start = 1.0
        T_end = 0.01
        alpha = 0.999 # Cooling rate
        T = T_start
        
        # We want to maximize size. 
        # Move: Remove one random point, add one random point (or just add if possible).
        # Actually, to change size, we need to be able to add points.
        # Standard SA on set size: 
        # 1. Try to add a random unselected point. If valid, accept (size+1).
        # 2. If no unselected point can be added, try a swap: remove one, add one.
        #    If the new set is valid, accept. Size stays same.
        #    This allows escaping local optima where no single addition is possible, but a swap is.
        
        # To actually increase size, we rely on the fact that after swaps, 
        # a previously blocked point might become addable.
        
        unselected = [i for i in range(num_points) if i not in current_set]
        
        while time.time() < deadline and T > T_end:
            # 1. Attempt to add a random unselected point
            added = False
            if unselected:
                idx_to_add = np.random.choice(unselected)
                # Check if adding idx_to_add is valid
                # We need to check against current_indices
                if is_valid_addition(np.array(current_indices), idx_to_add):
                    current_indices.append(idx_to_add)
                    current_set.add(idx_to_add)
                    unselected.remove(idx_to_add)
                    added = True
                    if len(current_indices) > best_size:
                        best_size = len(current_indices)
                        best_points = grid[np.array(current_indices)]
            
            # 2. If we didn't add, or even if we did, do a swap to explore
            # Let's do a swap to mix the state
            if len(current_indices) > 0:
                # Remove a random point
                idx_to_remove = np.random.choice(len(current_indices))
                removed_point = current_indices.pop(idx_to_remove)
                current_set.discard(removed_point)
                
                # Try to add a random unselected point (or the removed one back? No, pick new)
                # If unselected is empty, we can't swap.
                if unselected:
                    idx_to_add = np.random.choice(unselected)
                    # Check if adding idx_to_add is valid to the new set
                    if is_valid_addition(np.array(current_indices), idx_to_add):
                        current_indices.append(idx_to_add)
                        current_set.add(idx_to_add)
                        unselected.remove(idx_to_add)
                        # If we removed one and added one, size is same.
                        # But maybe we can now add more?
                        # Let's try to add another one immediately with some probability?
                        # No, let's keep it simple.
                    else:
                        # If the new point doesn't fit, put the old one back?
                        # Or accept the smaller set? 
                        # For SA, we usually accept worse states with probability exp(-delta/T).
                        # Here, "worse" means smaller size. 
                        # If we remove one and can't add one, size decreases by 1.
                        # This is bad for maximizing k. 
                        # Better strategy: Only swap if the new point fits. 
                        # If it doesn't fit, revert the removal.
                        current_indices.insert(idx_to_remove, removed_point)
                        current_set.add(removed_point)
                else:
                    # No unselected points, revert
                    current_indices.insert(idx_to_remove, removed_point)
                    current_set.add(removed_point)
            
            T *= alpha
            
            # If we got stuck (no additions in a while), maybe do a "shake"
            # For now, the swap mechanism should help.
            
            if time.time() > deadline - 5:
                break
                
        return best_points

    return solve()

# EVOLVE-BLOCK-START
# (The code above is the implementation)
# EVOLVE-BLOCK-END
