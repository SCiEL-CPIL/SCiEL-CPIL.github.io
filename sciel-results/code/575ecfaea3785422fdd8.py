import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points_total = len(grid)
    deadline = time.time() + 590
    
    # Precompute squared norms
    grid_sq = (grid ** 2).sum(axis=1)

    def is_valid_add(S_idx, p_idx):
        k = len(S_idx)
        if k < 4:
            return True
        
        S_pts = grid[S_idx]
        p = grid[p_idx]
        
        diff = S_pts - p
        sq_diff = grid_sq[S_idx] - grid_sq[p_idx]
        
        # Generate combinations of indices in S_idx (0 to k-1)
        combos = list(itertools.combinations(range(k), 4))
        if not combos:
            return True
            
        # Convert to numpy array for fast indexing
        combos_arr = np.array(combos, dtype=np.int32)
        
        # d_diff: (N, 4, 3)
        d_diff = diff[combos_arr]
        # d_sq: (N, 4)
        d_sq = sq_diff[combos_arr]
        
        # Combine to (N, 4, 4)
        mats = np.concatenate([d_diff, d_sq[:, :, None]], axis=2)
        
        dets = np.linalg.det(mats)
        
        return not np.any(np.abs(dets) < 0.5)

    # Initial Greedy with multiple restarts
    best_set = []
    np.random.seed(42)
    
    # We can try to find a good set quickly
    for _ in range(10):
        if time.time() > deadline: break
        
        order = np.random.permutation(num_points_total)
        curr_set = []
        
        for p_idx in order:
            if time.time() > deadline: break
            if is_valid_add(curr_set, p_idx):
                curr_set.append(p_idx)
        
        if len(curr_set) > len(best_set):
            best_set = curr_set.copy()

    # Local Search
    current_set = best_set.copy()
    in_set = np.zeros(num_points_total, dtype=bool)
    for idx in current_set:
        in_set[idx] = True
    
    # Helper to get random unused point
    def get_random_unused():
        unused = np.where(~in_set)[0]
        if len(unused) == 0:
            return -1
        return np.random.choice(unused)

    # Helper to get random used point
    def get_random_used():
        if not current_set:
            return -1
        return np.random.choice(current_set)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add
        for _ in range(50): # Try 50 random adds
            if time.time() > deadline: break
            p_idx = get_random_unused()
            if p_idx == -1: break
            if is_valid_add(current_set, p_idx):
                current_set.append(p_idx)
                in_set[p_idx] = True
                improved = True
                break # Restart add/swap loop
        
        if improved:
            continue

        # Try to swap
        for _ in range(200): # Try 200 random swaps
            if time.time() > deadline: break
            
            s_idx = get_random_used()
            if s_idx == -1: break
            p_idx = get_random_unused()
            if p_idx == -1: break
            
            # Check validity of (current_set - s_idx) + p_idx
            # We need to pass a list without s_idx
            # To avoid list creation cost, we can just pass the list and handle it?
            # No, is_valid_add uses len(S_idx).
            # Let's just create the list. It's small (max ~50).
            reduced_set = [x for x in current_set if x != s_idx]
            
            if is_valid_add(reduced_set, p_idx):
                # Perform swap
                current_set.remove(s_idx)
                current_set.append(p_idx)
                in_set[s_idx] = False
                in_set[p_idx] = True
                improved = True
                break # Restart loop

        # Optional: 2-swap if stuck?
        # If we are stuck for a while, try removing 2 and adding 2?
        # For now, simple 1-swap is good.

    # Final Polish: Try to add more points with a more systematic approach
    # This can sometimes find a few more points that random search missed
    for _ in range(10):
        if time.time() > deadline: break
        
        # Try adding points in a deterministic order (e.g., sorted by index)
        added_any = False
        for p_idx in range(num_points_total):
            if time.time() > deadline: break
            if not in_set[p_idx]:
                if is_valid_add(current_set, p_idx):
                    current_set.append(p_idx)
                    in_set[p_idx] = True
                    added_any = True
        
        if not added_any:
            break

    return grid[np.array(current_set)]
