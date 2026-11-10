import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    deadline = time.time() + 580  # Safety margin
    
    # Precompute squared norms for all grid points
    sq_norms = (grid ** 2).sum(axis=1).astype(np.float64)
    
    def is_valid_addition(arr: np.ndarray, p: np.ndarray) -> bool:
        """
        Check if adding point p to arr keeps the set valid.
        Early termination on first violating 4-tuple.
        """
        m = len(arr)
        if m < 4:
            return True
        
        p_idx = np.array(p, dtype=np.float64)
        p_sq = float(np.dot(p_idx, p_idx))
        
        # For each row i in arr: diff_i = arr[i] - p, dsq_i = |arr[i]|^2 - |p|^2
        # We need to check all 4-combinations of rows
        # Build the m x 4 matrix
        diff = arr.astype(np.float64) - p_idx
        dsq = sq_norms[:m] - p_sq  # arr is a subset of grid, but we need correct sq_norms
        
        # Actually, arr contains actual points from grid. Let's recompute properly.
        # arr is (m, 3) int64. p is (3,) int64 or float64.
        diff = arr.astype(np.float64) - p_idx
        dsq = (arr.astype(np.float64) ** 2).sum(axis=1) - p_sq
        
        M = np.empty((m, 4), dtype=np.float64)
        M[:, 0:3] = diff
        M[:, 3] = dsq
        
        # Check all 4-combinations with early exit
        for combo in itertools.combinations(range(m), 4):
            sub = M[list(combo)]
            d = np.linalg.det(sub)
            if abs(d) < 0.5:
                return False
        
        return True

    def get_random_order():
        return np.random.permutation(num_points_total)

    def greedy_build(order):
        curr_set = np.empty((0, 3), dtype=np.int64)
        for idx in order:
            if time.time() > deadline:
                break
            p = grid[idx]
            if is_valid_addition(curr_set, p):
                if len(curr_set) == 0:
                    curr_set = p[None, :]
                else:
                    curr_set = np.vstack([curr_set, p])
        return curr_set

    best_set = np.empty((0, 3), dtype=np.int64)
    
    np.random.seed(42)
    
    # Phase 1: Multiple Greedy Restarts
    num_restarts = 30
    for _ in range(num_restarts):
        if time.time() > deadline - 120:  # Reserve time for local search
            break
        order = get_random_order()
        curr_set = greedy_build(order)
        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()

    # Phase 2: Local Search
    current_set = best_set.copy()
    
    # Build tuple-to-index map for the grid
    grid_tuple_to_idx = {tuple(p): i for i, p in enumerate(grid)}
    
    current_indices = set()
    for p in current_set:
        current_indices.add(grid_tuple_to_idx[tuple(p)])
        
    all_indices = set(range(num_points_total))
    remaining_indices = list(all_indices - current_indices)
    
    no_improve_count = 0
    max_no_improve = 50
    consecutive_no_improve = 0
    
    while time.time() < deadline - 5:
        improved = False
        
        # 1. Try to Add - try all remaining points (not just a sample)
        if remaining_indices:
            # Shuffle and try in order; stop at first valid
            shuffled_remaining = list(remaining_indices)
            np.random.shuffle(shuffled_remaining)
            for idx in shuffled_remaining:
                if time.time() > deadline - 5:
                    break
                p = grid[idx]
                if is_valid_addition(current_set, p):
                    # Add
                    if len(current_set) == 0:
                        current_set = p[None, :]
                    else:
                        current_set = np.vstack([current_set, p])
                    
                    current_indices.add(idx)
                    remaining_indices.remove(idx)
                    improved = True
                    no_improve_count = 0
                    consecutive_no_improve = 0
                    break
        
        # 2. Try to Swap - more attempts
        if not improved and len(current_indices) > 0 and remaining_indices:
            swaps_to_try = 50
            for _ in range(swaps_to_try):
                if time.time() > deadline - 5:
                    break
                
                # Pick random in
                in_idx_val = list(current_indices)[np.random.randint(len(current_indices))]
                p_in = grid[in_idx_val]
                
                # Pick random out
                out_idx_val = remaining_indices[np.random.randint(len(remaining_indices))]
                p_out = grid[out_idx_val]
                
                # Find index of p_in in current_set
                in_pos = None
                for i in range(len(current_set)):
                    if np.array_equal(current_set[i], p_in):
                        in_pos = i
                        break
                
                if in_pos is None:
                    continue
                    
                reduced_mask = np.ones(len(current_set), dtype=bool)
                reduced_mask[in_pos] = False
                reduced_set = current_set[reduced_mask]
                
                if is_valid_addition(reduced_set, p_out):
                    # Perform swap
                    current_set = reduced_set
                    if len(current_set) == 0:
                        current_set = p_out[None, :]
                    else:
                        current_set = np.vstack([current_set, p_out])
                    
                    current_indices.remove(in_idx_val)
                    current_indices.add(out_idx_val)
                    remaining_indices.remove(out_idx_val)
                    remaining_indices.append(in_idx_val)
                    
                    improved = True
                    no_improve_count = 0
                    consecutive_no_improve = 0
                    break
        
        if not improved:
            no_improve_count += 1
            consecutive_no_improve += 1
            
            if no_improve_count > max_no_improve:
                # Kick: remove several points to escape local optimum
                if len(current_indices) > 8:
                    kick_size = min(3 + consecutive_no_improve // 10, max(2, len(current_indices) // 3))
                    kick_size = min(kick_size, len(current_indices) - 3)  # Keep at least 3
                    if kick_size > 0:
                        kick_idxs = np.random.choice(list(current_indices), kick_size, replace=False)
                        for idx in kick_idxs:
                            p = grid[idx]
                            mask = np.ones(len(current_set), dtype=bool)
                            for i in range(len(current_set)):
                                if np.array_equal(current_set[i], p):
                                    mask[i] = False
                                    break
                            current_set = current_set[mask]
                            
                            current_indices.remove(idx)
                            remaining_indices.append(idx)
                        no_improve_count = 0
                        max_no_improve = min(max_no_improve + 20, 200)

    return current_set.astype(np.int64)
