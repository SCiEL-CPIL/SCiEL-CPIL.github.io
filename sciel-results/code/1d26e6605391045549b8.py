import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    deadline = time.time() + 580  # Safety margin
    
    def is_valid_addition(arr: np.ndarray, p: np.ndarray) -> bool:
        """
        Check if adding point p to arr keeps the set valid.
        Early-terminating determinant check.
        """
        m = len(arr)
        if m < 4:
            return True
        
        # diff[i] = arr[i] - p
        diff = arr - p
        # sq_diff[i] = |arr[i]|^2 - |p|^2
        sq_diff = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        
        # Construct the (m, 4) matrix M where rows are (dx, dy, dz, dsq)
        M = np.empty((m, 4), dtype=np.float64)
        M[:, 0:3] = diff
        M[:, 3] = sq_diff
        
        # For small m, compute all at once
        if m <= 20:
            combos = list(itertools.combinations(range(m), 4))
            if not combos:
                return True
            combo_arr = np.array(combos, dtype=np.int32)
            M_sub = M[combo_arr]
            dets = np.linalg.det(M_sub)
            return not np.any(np.abs(dets) < 0.5)
        
        # For larger m, check in batches with early termination
        # Process combinations in chunks to allow early exit
        chunk_size = 5000
        combos = itertools.combinations(range(m), 4)
        
        for i in range(0, 1, chunk_size):  # placeholder to structure
            pass
        
        # Better: iterate with early termination
        count = 0
        for combo in itertools.combinations(range(m), 4):
            M_sub = M[list(combo)]
            det_val = np.linalg.det(M_sub)
            if abs(det_val) < 0.5:
                return False
            count += 1
            # For very large m, this is still slow. Let's batch it.
            # Actually, let's just do batched processing
            break
        
        # Batched approach for efficiency
        remaining_combos = []
        # We already checked the first one if m > 20, let's redo properly
        # Actually, let me restructure: for m > 20, use batched numpy
        
        # Collect combos in batches
        combo_list = []
        for c in itertools.combinations(range(m), 4):
            combo_list.append(c)
            if len(combo_list) >= chunk_size:
                combo_arr = np.array(combo_list, dtype=np.int32)
                M_sub = M[combo_arr]
                dets = np.linalg.det(M_sub)
                if np.any(np.abs(dets) < 0.5):
                    return False
                combo_list = []
        
        if combo_list:
            combo_arr = np.array(combo_list, dtype=np.int32)
            M_sub = M[combo_arr]
            dets = np.linalg.det(M_sub)
            if np.any(np.abs(dets) < 0.5):
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
        if time.time() > deadline:
            break
        order = get_random_order()
        curr_set = greedy_build(order)
        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()

    # Phase 2: Local Search
    current_set = best_set.copy()
    
    # Build a tuple-to-index map for the grid
    grid_tuple_to_idx = {tuple(p): i for i, p in enumerate(grid)}
    
    current_indices = set()
    for p in current_set:
        current_indices.add(grid_tuple_to_idx[tuple(p)])
        
    all_indices = set(range(num_points_total))
    remaining_indices = list(all_indices - current_indices)
    
    # Track points that were recently rejected for addition (for smarter swaps)
    rejected_points = []  # list of (idx, rejection_count)
    rejection_counts = {}  # idx -> count of times rejected
    
    no_improve_count = 0
    max_no_improve = 150
    
    # Build position lookup for current_set
    def get_pos_in_set(p_tuple):
        for i in range(len(current_set)):
            if tuple(current_set[i]) == p_tuple:
                return i
        return -1
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to Add
        if remaining_indices:
            # Prioritize points with lower rejection counts
            scored_candidates = []
            for idx in remaining_indices:
                rc = rejection_counts.get(idx, 0)
                scored_candidates.append((rc, idx))
            scored_candidates.sort()
            
            # Take top candidates (lowest rejection count first)
            top_candidates = [x[1] for x in scored_candidates[:min(100, len(scored_candidates))]]
            # Add some random ones too
            if len(remaining_indices) > len(top_candidates):
                extra = set(remaining_indices) - set(top_candidates)
                if extra:
                    extra_list = list(extra)
                    num_extra = min(20, len(extra_list))
                    top_candidates.extend(np.random.choice(extra_list, num_extra, replace=False).tolist())
            
            np.random.shuffle(top_candidates)
            
            for idx in top_candidates:
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
                    break
                else:
                    rejection_counts[idx] = rejection_counts.get(idx, 0) + 1
        
        # 2. Try to Swap
        if not improved and len(current_indices) > 0 and remaining_indices:
            swaps_to_try = 30
            for _ in range(swaps_to_try):
                if time.time() > deadline:
                    break
                
                # Pick random in
                in_idx_val = list(current_indices)[np.random.randint(len(current_indices))]
                p_in = grid[in_idx_val]
                
                # Pick out: prioritize points with high rejection counts (more likely to fit if we remove a blocker)
                scored_out = []
                for idx in remaining_indices:
                    rc = rejection_counts.get(idx, 0)
                    scored_out.append((rc, idx))
                scored_out.sort(reverse=True)  # highest rejection first
                
                # Mix of high-rejection and random
                top_out = [x[1] for x in scored_out[:min(20, len(scored_out))]]
                if len(remaining_indices) > len(top_out):
                    extra = set(remaining_indices) - set(top_out)
                    if extra:
                        extra_list = list(extra)
                        num_extra = min(10, len(extra_list))
                        top_out.extend(np.random.choice(extra_list, num_extra, replace=False).tolist())
                
                out_idx_val = top_out[np.random.randint(len(top_out))]
                p_out = grid[out_idx_val]
                
                # Find index of p_in in current_set
                in_pos = get_pos_in_set(tuple(p_in))
                if in_pos == -1:
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
                    
                    # Reset rejection count for the newly added point
                    rejection_counts[out_idx_val] = 0
                    # Keep rejection count for the removed point
                    rejection_counts[in_idx_val] = rejection_counts.get(in_idx_val, 0) + 1
                    
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Kick: remove multiple points to escape local optimum
                if len(current_indices) > 10:
                    kick_size = min(7, max(3, len(current_indices) // 5))
                    # Remove random points
                    kick_idxs = np.random.choice(list(current_indices), kick_size, replace=False)
                    for idx in kick_idxs:
                        p = grid[idx]
                        # Remove from current_set
                        pos = get_pos_in_set(tuple(p))
                        if pos != -1:
                            mask = np.ones(len(current_set), dtype=bool)
                            mask[pos] = False
                            current_set = current_set[mask]
                        
                        current_indices.remove(idx)
                        remaining_indices.append(idx)
                        rejection_counts[idx] = 0
                    no_improve_count = 0
                    max_no_improve = 250

    return current_set.astype(np.int64)
