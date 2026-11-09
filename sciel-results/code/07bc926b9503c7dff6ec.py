import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    
    # Pre-compute squared norms
    sq_norms = (grid ** 2).sum(axis=1)
    
    def can_add(S_indices: np.ndarray, p_idx: int) -> bool:
        """
        Check if grid[p_idx] can be added to the set of points S_indices.
        S_indices is an array of indices into grid.
        """
        m = len(S_indices)
        if m < 4:
            return True
            
        p = grid[p_idx]
        sq_p = sq_norms[p_idx]
        
        # Get coordinates of points in S
        S_pts = grid[S_indices]
        sq_S = sq_norms[S_indices]
        
        # Form vectors (q - p, |q|^2 - |p|^2)
        # Shape (m, 4)
        V = np.empty((m, 4), dtype=np.float64)
        V[:, 0:3] = S_pts - p
        V[:, 3] = sq_S - sq_p
        
        # Check all combinations of 4
        # If m is large, this is the bottleneck.
        combos = np.array(list(itertools.combinations(range(m), 4)))
        if len(combos) == 0:
            return True
            
        sub = V[combos] # (C, 4, 4)
        dets = np.linalg.det(sub)
        
        return not np.any(np.abs(dets) < 0.5)

    def get_bad_subsets(S_indices: np.ndarray, p_idx: int) -> list:
        """
        Returns a list of 4-tuples of indices in S_indices that cause the conflict with p_idx.
        """
        m = len(S_indices)
        if m < 4:
            return []
            
        p = grid[p_idx]
        sq_p = sq_norms[p_idx]
        
        S_pts = grid[S_indices]
        sq_S = sq_norms[S_indices]
        
        V = np.empty((m, 4), dtype=np.float64)
        V[:, 0:3] = S_pts - p
        V[:, 3] = sq_S - sq_p
        
        combos = np.array(list(itertools.combinations(range(m), 4)))
        sub = V[combos]
        dets = np.linalg.det(sub)
        
        bad_mask = np.abs(dets) < 0.5
        bad_combos = combos[bad_mask]
        
        # Map back to original indices in S_indices
        # combos contains indices relative to S_indices
        # We want indices relative to S_indices to pick one to remove
        return bad_combos

    def random_greedy(seed: int) -> np.ndarray:
        np.random.seed(seed)
        order = np.random.permutation(num_points)
        S = np.empty(0, dtype=np.int64)
        for p_idx in order:
            if can_add(S, p_idx):
                S = np.append(S, p_idx)
        return S

    start_time = time.time()
    timeout = 580.0 # Safety margin
    
    best_S = None
    best_size = 0
    
    # Phase 1: Multiple Random Greedy Restarts
    # We have 600s. Greedy is fast. Let's do many restarts.
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() - start_time > timeout:
            break
        S = random_greedy(i)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            
    if best_S is None:
        return np.empty((0, 3), dtype=np.int64)
        
    # Phase 2: Local Search (Add & Swap)
    # We maintain a set S.
    # We try to add points. If we can't, we try swaps.
    
    S = best_S.copy()
    S_set = set(S.tolist())
    
    # Pre-compute list of points not in S
    # We will iterate through them.
    
    while time.time() - start_time < timeout:
        added_any = False
        
        # Try to add any point not in S
        candidates = [i for i in range(num_points) if i not in S_set]
        np.random.shuffle(candidates)
        
        for p_idx in candidates:
            if can_add(S, p_idx):
                S = np.append(S, p_idx)
                S_set.add(p_idx)
                added_any = True
                # Restart candidate loop? Or just continue.
                # Continuing is fine, we might add more.
                break # Restart to re-evaluate "candidates" list order? 
                # Actually, just breaking out of the inner loop and restarting the outer while is cleaner.
        
        if added_any:
            continue
            
        # If we couldn't add any point, try swaps
        # Find a point p_in not in S that is blocked.
        # Pick a random p_in not in S.
        if not candidates:
            break
            
        p_in = candidates[0]
        bad_subsets = get_bad_subsets(S, p_in)
        
        if len(bad_subsets) == 0:
            # Should not happen if can_add returned False
            break
            
        # Try to remove one point from one of the bad subsets
        # Pick a random bad subset
        bad_combo = bad_subsets[np.random.randint(len(bad_subsets))]
        # Pick a random point from this subset to remove
        p_out_idx_in_S = bad_combo[np.random.randint(4)]
        p_out = S[p_out_idx_in_S]
        
        # Remove p_out and try to add p_in
        S_new = np.delete(S, p_out_idx_in_S)
        S_new_set = S_set.copy()
        S_new_set.remove(p_out)
        
        if can_add(S_new, p_in):
            # Swap is valid
            S = np.append(S_new, p_in)
            S_new_set.add(p_in)
            S_set = S_new_set
            # Don't update best_size immediately, but the size is the same.
            # We might be able to add more later.
            
        # To avoid cycling, we could keep a history, but for now, randomization helps.
        # We can also try multiple swaps per iteration if time permits.
        
    # Final check and return
    # Verify validity of best_S
    # The problem asks for the points, not indices.
    if best_S is not None and len(best_S) > len(S):
        final_indices = best_S
    else:
        final_indices = S
        
    # Convert indices to coordinates
    return grid[final_indices].astype(np.int64)
