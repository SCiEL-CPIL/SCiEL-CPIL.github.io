# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Finds a large subset of the 8x8x8 grid such that no 5 points are co-spherical.
    Uses a multi-start greedy algorithm followed by a local search (swap) phase
    to escape local maxima.
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute squared norms
    sq_norms = (grid ** 2).sum(axis=1)
    
    # Precompute the lifted coordinates relative to each point? No, too big.
    # We compute on the fly.
    
    def is_valid_set(points_idx):
        """
        Checks if the set of points (indices) is valid.
        Returns True if no 5 points are co-spherical.
        """
        m = len(points_idx)
        if m < 5:
            return True
        
        # We need to check all 5-tuples?
        # It is faster to check incrementally or use the 4-relative check.
        # For a full set check, we can iterate through points and check against previous 4.
        # This is O(m^5) if done naively with combinations.
        # Better: For each i, check if point i is compatible with the set of previous points?
        # No, compatibility is not transitive.
        
        # Let's use the property: A set is valid iff for every point p in the set,
        # and every 4 other points q1..q4 in the set, p, q1..q4 are not co-spherical.
        # This is redundant. We just need to check all 5-tuples.
        
        # Optimization: 
        # If we build the set incrementally, we only check the new point against the old ones.
        # For a full re-check (needed for swaps), we might need to be careful.
        # However, if S is valid, and we remove u and add v, we only need to check:
        # 1. Tuples containing v and 4 points from S\{u}.
        # 2. Tuples containing u are gone, so no need to check them.
        # 3. Tuples not containing u or v are already valid.
        # So `can_add(v, S\{u})` is sufficient.
        
        pass # Not used directly in the main loop, logic is in can_add

    def can_add(arr_idx, p_idx):
        """
        Checks if p_idx can be added to the set of points with indices arr_idx.
        arr_idx: list of indices into grid
        p_idx: index into grid
        """
        m = len(arr_idx)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = sq_norms[p_idx]
        
        # Get existing points
        # arr_idx is a list of indices
        # We need to check all 4-combinations of arr_idx
        if m == 4:
            q1, q2, q3, q4 = [grid[i] for i in arr_idx]
            # Build 4x4 matrix
            # Rows: (q - p, |q|^2 - |p|^2)
            # q1 - p, ...
            v1 = q1 - p; w1 = sq_norms[arr_idx[0]] - p_sq
            v2 = q2 - p; w2 = sq_norms[arr_idx[1]] - p_sq
            v3 = q3 - p; w3 = sq_norms[arr_idx[2]] - p_sq
            v4 = q4 - p; w4 = sq_norms[arr_idx[3]] - p_sq
            
            M = np.array([
                [v1[0], v1[1], v1[2], w1],
                [v2[0], v2[1], v2[2], w2],
                [v3[0], v3[1], v3[2], w3],
                [v4[0], v4[1], v4[2], w4]
            ])
            return abs(np.linalg.det(M)) >= 0.5
        
        # If m > 4, we need to check all 4-combinations
        # This is expensive. 
        # Optimization: If m is large, we can sample? No, must be exact.
        # We can use itertools.combinations
        for combo in itertools.combinations(range(m), 4):
            idxs = [arr_idx[i] for i in combo]
            q1, q2, q3, q4 = [grid[i] for i in idxs]
            
            v1 = q1 - p; w1 = sq_norms[idxs[0]] - p_sq
            v2 = q2 - p; w2 = sq_norms[idxs[1]] - p_sq
            v3 = q3 - p; w3 = sq_norms[idxs[2]] - p_sq
            v4 = q4 - p; w4 = sq_norms[idxs[3]] - p_sq
            
            M = np.array([
                [v1[0], v1[1], v1[2], w1],
                [v2[0], v2[1], v2[2], w2],
                [v3[0], v3[1], v3[2], w3],
                [v4[0], v4[1], v4[2], w4]
            ])
            if abs(np.linalg.det(M)) < 0.5:
                return False
        return True

    def run_greedy_and_search(seed, time_limit=580):
        np.random.seed(seed)
        start_time = time.time()
        
        # Greedy Phase
        order = np.random.permutation(N)
        current_set = []
        current_set_set = set()
        
        for p_idx in order:
            if time.time() - start_time > time_limit:
                break
            if can_add(current_set, p_idx):
                current_set.append(p_idx)
                current_set_set.add(p_idx)
        
        # Local Search Phase (Swaps)
        # We try to swap points to improve "slack" or just to allow more additions later.
        # Since we want to maximize size, and greedy gets stuck, swaps can rearrange the set
        # to a "less dense" configuration that might accept more points.
        
        best_size = len(current_set)
        best_set = list(current_set)
        
        # We will do a number of iterations
        max_iters = 500
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Try to add a random point
            added = False
            for _ in range(10): # Try a few random additions
                p_idx = np.random.randint(N)
                if p_idx not in current_set_set:
                    if can_add(current_set, p_idx):
                        current_set.append(p_idx)
                        current_set_set.add(p_idx)
                        added = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = list(current_set)
                        break
            
            if not added:
                # Try a swap
                if len(current_set) > 0:
                    u_idx_pos = np.random.randint(len(current_set))
                    u_idx = current_set[u_idx_pos]
                    
                    v_idx = np.random.randint(N)
                    if v_idx not in current_set_set:
                        # Check if swap is valid
                        # New set would be current_set without u_idx, plus v_idx
                        # We only need to check if v_idx is compatible with current_set \ {u_idx}
                        
                        # Create a list without u_idx
                        temp_set = [x for i, x in enumerate(current_set) if i != u_idx_pos]
                        
                        if can_add(temp_set, v_idx):
                            # Perform swap
                            current_set[u_idx_pos] = v_idx
                            current_set_set.remove(u_idx)
                            current_set_set.add(v_idx)
                            # Note: Size doesn't change, but structure does.
                            # This might allow future additions.
            
        return np.array([grid[i] for i in best_set], dtype=np.int64)

    np.random.seed(0)
    best_overall = None
    best_size_overall = 0
    
    # Run multiple restarts
    num_restarts = 10
    for i in range(num_restarts):
        res = run_greedy_and_search(seed=i, time_limit=580)
        if len(res) > best_size_overall:
            best_size_overall = len(res)
            best_overall = res
            
        if time.time() > 550: # Safety margin
            break
            
    return best_overall.astype(np.int64)

# EVOLVE-BLOCK-END
