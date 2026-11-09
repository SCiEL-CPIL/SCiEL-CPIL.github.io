import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1)
    
    # Precompute combinations for efficiency? 
    # No, m changes.
    
    def can_add(arr_idx, p_idx):
        m = len(arr_idx)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = grid_sq[p_idx]
        arr_pts = grid[arr_idx]
        arr_sq = grid_sq[arr_idx]
        
        diff = arr_pts - p
        sq_diff = arr_sq - p_sq
        
        # Get all 4-combinations of indices
        # If m is large, this list is big.
        # For m=40, ~90k combos.
        # We can generate indices as a numpy array of shape (C, 4)
        combos = np.array(list(itertools.combinations(range(m), 4)))
        if len(combos) == 0:
            return True
            
        sub_diff = diff[combos] # (C, 4, 3)
        sub_sq = sq_diff[combos] # (C, 4)
        
        # Construct matrices
        # Rows: q1-p, q2-p, q3-p, q4-p
        # Cols: x, y, z, sq_diff
        mat = np.concatenate([sub_diff, sub_sq[:, :, None]], axis=2) # (C, 4, 4)
        
        dets = np.linalg.det(mat)
        
        return not np.any(np.abs(dets) < 0.5)

    def greedy_fill(order):
        selected = []
        for idx in order:
            if can_add(np.array(selected), idx):
                selected.append(idx)
        return np.array(selected)

    def local_search(selected_arr, time_limit):
        # selected_arr is array of indices
        current = list(selected_arr)
        current_set = set(current)
        start_time = time.time()
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try to add
            for idx in range(N):
                if idx in current_set:
                    continue
                if can_add(np.array(current), idx):
                    current.append(idx)
                    current_set.add(idx)
                    improved = True
                    break # Restart add phase
            
            if improved:
                continue
            
            # 2. Try to swap (perturb)
            # Pick a random point to remove
            if len(current) > 0:
                out_idx_pos = np.random.randint(len(current))
                out_idx = current[out_idx_pos]
                
                # Try to find a valid replacement
                # Shuffle candidates to avoid bias
                candidates = [i for i in range(N) if i not in current_set]
                np.random.shuffle(candidates)
                
                swapped = False
                for in_idx in candidates:
                    # Check if swapping out_idx with in_idx is valid
                    # Remove out_idx, add in_idx
                    temp_current = current.copy()
                    temp_current.pop(out_idx_pos)
                    if can_add(np.array(temp_current), in_idx):
                        # Accept swap
                        current[out_idx_pos] = in_idx
                        current_set.remove(out_idx)
                        current_set.add(in_idx)
                        swapped = True
                        break
                
                if not swapped:
                    # If we can't swap this point, try another? 
                    # Or break to avoid infinite loop if no swaps possible?
                    # If no swaps possible and no adds possible, we are stuck.
                    # We should break and restart or just stop.
                    # Let's try a few more random removals before giving up.
                    # For now, break to save time.
                    break

        return np.array(current)

    best_size = 0
    best_arr = np.array([], dtype=np.int64)
    
    start_time = time.time()
    deadline = start_time + 630 - 20 # Safety margin
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Generate order
        # Mix of random and heuristic
        r = rng.random()
        if r < 0.3:
            order = rng.permutation(N)
        elif r < 0.6:
            # Sort by distance from center
            center = np.array([3.5, 3.5, 3.5])
            dists = np.linalg.norm(grid - center, axis=1)
            noise = rng.random(N) * 2.0
            order = np.argsort(dists + noise)
        else:
            # Sort by sum of coords
            sums = grid.sum(axis=1)
            noise = rng.random(N) * 2.0
            order = np.argsort(sums + noise)
            
        current = greedy_fill(order)
        
        # Local search
        # Allocate remaining time proportionally or fixed chunk
        # Let's give it a chunk of time
        ls_deadline = time.time() + min(10, deadline - time.time())
        current = local_search(current, ls_deadline)
        
        if len(current) > len(best_arr):
            best_arr = current.copy()
            best_size = len(current)
            
            # If we found a new best, maybe do more local search on it?
            # The loop will restart and potentially find it again or improve.
            # To be efficient, we can try to extend the best solution found so far.
            # But restarting is simpler and robust.
            
    return grid[best_arr]
