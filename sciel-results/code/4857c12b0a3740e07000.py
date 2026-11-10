import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Selects points from an 8x8x8 grid such that no 5 points are co-spherical (or co-planar).
    
    Strategy:
    1. Generate a dense initial set using a fast, randomized greedy algorithm.
    2. Use local search (swaps) to optimize the set size, ensuring validity at every step.
    """
    n = 8
    deadline = time.time() + 580  # Safety margin
    
    # Generate all grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute squared norms
    sq_norms_grid = (grid ** 2).sum(axis=1)
    
    def is_valid_set(indices):
        """
        Check if a set of points is valid (no 5 co-spherical).
        Returns True if valid.
        """
        if len(indices) < 5:
            return True
        
        idx_arr = np.array(indices)
        pts = grid[idx_arr]
        norms = sq_norms_grid[idx_arr]
        k = len(idx_arr)
        
        # For each point p, check all 4-subsets of the other points
        for i in range(k):
            p_idx = idx_arr[i]
            p_pts = grid[p_idx]
            p_norm = norms[i]
            
            # Indices of other points
            others_mask = np.ones(k, dtype=bool)
            others_mask[i] = False
            others_idx = np.where(others_mask)[0]
            
            if len(others_idx) < 4:
                continue
            
            o_pts = grid[idx_arr[others_idx]]
            o_norms = norms[others_idx]
            
            diff = o_pts - p_pts
            r = o_norms - p_norm
            
            W = np.column_stack((diff, r))
            
            m = len(others_idx)
            # Generate all 4-combinations of indices in [0, m)
            # For efficiency, we can vectorize the determinant calculation
            # But itertools.combinations is standard.
            # C(m, 4) for m~50 is ~230k, which is fast with vectorized det.
            
            # Optimization: If m is large, this is the bottleneck.
            # We can check a random subset first? No, correctness is key.
            
            # We'll generate the combinations and compute dets.
            # To avoid memory issues with very large sets, we can process in chunks,
            # but for k<100, C(k,4) is manageable.
            
            # However, generating the list of combinations in Python is slow.
            # Let's try to optimize the check.
            
            # Alternative:
            # The condition is that for ANY 5 points, the determinant is not 0.
            # We are checking if ANY 5-point subset has det < 0.5.
            
            # Let's just use itertools. It's C-implemented.
            comb_indices = list(itertools.combinations(range(m), 4))
            if not comb_indices:
                continue
                
            # Convert to numpy array for fancy indexing
            # This creates a (C(m,4), 4) array
            # Then W[comb_indices] creates (C(m,4), 4, 4)
            # np.linalg.det can handle this.
            
            # To speed up, we can check if the set is "obviously" bad?
            # No, just compute.
            
            # Optimization: Only check if we haven't already failed.
            # We break on first failure.
            
            # Vectorized check
            # Note: W is (m, 4). We need to select rows.
            # comb_indices is a list of tuples.
            
            # This might be slow for large m.
            # Let's see if we can early exit.
            
            # For now, let's just do it.
            # If k is around 40-50, C(49,4) is ~211k.
            # 211k 4x4 dets is very fast in numpy.
            
            subs = W[np.array(comb_indices)]
            dets = np.linalg.det(subs)
            
            if np.any(np.abs(dets) < 0.5):
                return False
                
        return True

    def fast_greedy(seed_val, initial_set=None):
        """
        Greedy construction of a valid set.
        """
        rng = np.random.default_rng(seed_val)
        
        if initial_set is None:
            current_set = []
            in_set = np.zeros(N, dtype=bool)
        else:
            current_set = list(initial_set)
            in_set = np.zeros(N, dtype=bool)
            in_set[np.array(current_set)] = True
            
        # Shuffle order of all points
        order = rng.permutation(N)
        
        for p_idx in order:
            if in_set[p_idx]:
                continue
                
            # Try to add p_idx
            new_set = current_set + [p_idx]
            
            # Check validity
            # Optimization: We only need to check if p_idx violates with existing 4 points.
            # Since current_set is valid, any violation must involve p_idx.
            # So we only need to check 4-subsets of current_set that include p_idx?
            # No, the test checks "for each point p with at least 4 earlier points...".
            # If we add p_idx, it becomes the "later" point.
            # So we need to check all 4-subsets of current_set.
            
            if len(current_set) < 4:
                current_set.append(p_idx)
                in_set[p_idx] = True
                continue
            
            # Check if p_idx can be added
            # We need to check all 4-subsets of current_set.
            # Let's reuse the logic from is_valid_set but optimized for adding one point.
            
            # Construct the check for p_idx against current_set
            arr_pts = grid[np.array(current_set)]
            p_pts = grid[p_idx]
            
            diff = arr_pts - p_pts
            r = sq_norms_grid[np.array(current_set)] - sq_norms_grid[p_idx]
            
            W = np.column_stack((diff, r))
            
            m = len(current_set)
            if m < 4:
                current_set.append(p_idx)
                in_set[p_idx] = True
                continue
                
            comb_indices = list(itertools.combinations(range(m), 4))
            if not comb_indices:
                current_set.append(p_idx)
                in_set[p_idx] = True
                continue
                
            subs = W[np.array(comb_indices)]
            dets = np.linalg.det(subs)
            
            if not np.any(np.abs(dets) < 0.5):
                current_set.append(p_idx)
                in_set[p_idx] = True
            
            if time.time() > deadline - 10:
                break
                
        return current_set

    # --- Phase 1: Greedy Initialization ---
    best_set = []
    
    # Run multiple greedy restarts
    for i in range(10):
        if time.time() > deadline - 100:
            break
        cand = fast_greedy(42 + i)
        if len(cand) > len(best_set):
            best_set = cand
            
    if not best_set:
        return np.empty((0, 3), dtype=np.int64)
        
    # --- Phase 2: Local Search ---
    # Try to improve the set by swapping points.
    # Goal: Maximize size.
    
    current_set = list(best_set)
    in_set = np.zeros(N, dtype=bool)
    in_set[np.array(current_set)] = True
    
    # We will try to add points. If stuck, try swaps.
    
    # Helper to find addable point
    def find_addable_point(set_list):
        set_arr = np.array(set_list)
        in_set_local = np.zeros(N, dtype=bool)
        in_set_local[set_arr] = True
        candidates = np.where(~in_set_local)[0]
        
        # Randomize
        rng = np.random.default_rng()
        rng.shuffle(candidates)
        
        for p_idx in candidates:
            # Check if p_idx can be added
            if len(set_list) < 4:
                return p_idx
                
            arr_pts = grid[np.array(set_list)]
            p_pts = grid[p_idx]
            
            diff = arr_pts - p_pts
            r = sq_norms_grid[np.array(set_list)] - sq_norms_grid[p_idx]
            
            W = np.column_stack((diff, r))
            
            m = len(set_list)
            comb_indices = list(itertools.combinations(range(m), 4))
            if not comb_indices:
                return p_idx
                
            subs = W[np.array(comb_indices)]
            dets = np.linalg.det(subs)
            
            if not np.any(np.abs(dets) < 0.5):
                return p_idx
                
        return None

    # Main loop
    while time.time() < deadline - 10:
        
        # 1. Try to add a point
        add_idx = find_addable_point(current_set)
        if add_idx is not None:
            current_set.append(add_idx)
            in_set[add_idx] = True
            if len(current_set) > len(best_set):
                best_set = list(current_set)
            continue
            
        # 2. Stuck. Try swaps.
        # Pick a point in set to remove, a point outside to add.
        # We want to find a swap that allows adding more points later?
        # Or just a swap that keeps the set valid and maybe allows a different add?
        
        # Simple strategy: Random swaps.
        # If a swap results in a valid set, accept it.
        # Then try to add again.
        
        found_swap = False
        # Limit swap attempts to avoid wasting time
        for _ in range(50):
            if time.time() > deadline - 10:
                break
                
            if len(current_set) == 0:
                break
                
            # Pick random point to remove
            rem_pos = np.random.randint(0, len(current_set))
            rem_idx = current_set[rem_pos]
            
            # Pick random point to add
            candidates = np.where(~in_set)[0]
            if len(candidates) == 0:
                break
            add_idx = candidates[np.random.randint(0, len(candidates))]
            
            # Perform swap
            new_set = list(current_set)
            new_set.pop(rem_pos)
            new_set.append(add_idx)
            
            # Check validity
            if is_valid_set(new_set):
                # Accept swap
                in_set[rem_idx] = False
                in_set[add_idx] = True
                current_set = new_set
                found_swap = True
                break
                
        if not found_swap:
            # No swap found. Maybe restart greedy from a subset?
            # Or just restart the whole greedy process?
            
            # Let's try a "shake": remove a random point, then greedily add.
            if len(current_set) > 0:
                rem_pos = np.random.randint(0, len(current_set))
                rem_idx = current_set[rem_pos]
                current_set.pop(rem_pos)
                in_set[rem_idx] = False
                
                # Greedily add points
                new_set = fast_greedy(np.random.randint(0, 10000), current_set)
                if len(new_set) > len(current_set):
                    # Update in_set
                    in_set = np.zeros(N, dtype=bool)
                    in_set[np.array(new_set)] = True
                    current_set = new_set
                    if len(current_set) > len(best_set):
                        best_set = list(current_set)

    # Final validation
    if not is_valid_set(best_set):
        # Fallback: find largest valid prefix?
        # Or just return empty.
        # Let's try to fix it by removing points until valid.
        # But this might be slow.
        # Given the logic, best_set should be valid.
        # If not, return empty to be safe.
        return np.empty((0, 3), dtype=np.int64)
        
    return grid[np.array(best_set)]
