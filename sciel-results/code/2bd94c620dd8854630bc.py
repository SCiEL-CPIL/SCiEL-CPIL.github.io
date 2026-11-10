# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Returns a 2-D array of shape (k, 11) with integer entries (int64) such that:
      - second dimension is 11
      - k >= 2
      - no point is the zero vector
      - min squared pairwise distance >= max squared norm
    
    Strategy:
    1. Generate pool of weight-w vectors for w=4.
       For w=4: Norm^2=4, Constraint dot <= 2. Pool size 5280.
       
    2. Use a fast greedy selection with bitset conflict masks for initialization.
       Precompute conflict matrix.
       
    3. Apply local search (1-swap) to improve the greedy solution.
       The local search tries to swap out one selected point for one unselected point
       if it results in a valid configuration. This helps escape local optima of the greedy method.
       
    4. Repeat with different random seeds.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin (evaluator kills at 630s)
    d = 11
    
    def generate_pool(w):
        pool = []
        for idx in itertools.combinations(range(d), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = np.zeros(d, dtype=np.int64)
                for i, s in zip(idx, signs):
                    v[i] = s
                pool.append(v)
        return np.array(pool)
    
    def greedy_init(pool, conflict_matrix, order):
        """Fast greedy initialization using precomputed conflict matrix."""
        n = len(pool)
        available = np.ones(n, dtype=bool)
        selected = []
        
        for idx in order:
            if not available[idx]:
                continue
            selected.append(idx)
            available[conflict_matrix[idx]] = False
            
        return selected
    
    def local_search(pool, conflict_matrix, selected, time_limit, max_iters=1000):
        """
        Local search improvement: try to swap one selected point with one unselected point.
        """
        n = len(pool)
        selected_set = set(selected)
        unselected = [i for i in range(n) if i not in selected_set]
        
        current_size = len(selected)
        best_selected = selected[:]
        best_size = current_size
        
        # Precompute degrees or just iterate
        # To speed up, we can randomly sample unselected points or iterate through them.
        # Since N is ~5280, iterating all is O(N). Checking compatibility is O(k).
        # Total O(N*k) per full pass. k~140, N~5000 -> ~700k ops. Fast enough for 1000 iters?
        # 700M ops might be slow in Python. Let's be careful.
        
        # Optimization: Precompute for each unselected point, which selected points it conflicts with?
        # That's expensive to maintain.
        
        # Alternative: Just try random swaps.
        # Or iterate through selected points and find a compatible replacement.
        
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < time_limit:
            improved = False
            iters += 1
            
            # Shuffle unselected to randomize
            np.random.shuffle(unselected)
            
            for i in range(len(selected)):
                if time.time() > time_limit:
                    break
                
                sel_idx = selected[i]
                
                # Try to find a candidate to swap with sel_idx
                # Candidate c is valid if:
                # 1. c is not in selected
                # 2. c is compatible with all selected points EXCEPT sel_idx
                
                # We can check this by:
                # compatible_with_all = True
                # for j in selected:
                #    if j == sel_idx: continue
                #    if conflict_matrix[c, j]: compatible_with_all = False; break
                # if compatible_with_all: swap
                
                # To speed up, we can precompute the "conflict count" of each unselected point with the current selected set.
                # But that's hard to update incrementally in a simple loop.
                
                # Let's just iterate through a subset of unselected points to avoid O(N*k) per swap attempt if N is large.
                # Or just iterate all, it's 5000 * 140 ~ 700k checks. In Python, ~0.05s.
                # 1000 iters * 140 swaps/iter (worst case) * 0.05s = 7000s. Too slow.
                
                # We need a faster check.
                # Use numpy vectorized check?
                # conflicts = conflict_matrix[unselected, :].sum(axis=1) ? No, that's total conflicts.
                # We need conflicts with (selected - {sel_idx}).
                
                # Let's maintain a bitmask or just use the matrix.
                # Actually, let's just try a few random candidates per iteration.
                
                # Pick a random candidate from unselected
                if len(unselected) == 0:
                    break
                
                cand = unselected[np.random.randint(len(unselected))]
                
                # Check if cand is compatible with all selected except sel_idx
                # selected_except_i = [s for j, s in enumerate(selected) if j != i]
                # if all(not conflict_matrix[cand, s] for s in selected_except_i):
                
                # Vectorized check:
                # mask = np.ones(n, dtype=bool)
                # mask[selected] = False
                # mask[sel_idx] = True # We allow conflict with sel_idx
                # # Wait, we need to check if cand conflicts with any selected point other than sel_idx.
                # # conflict_matrix[cand, :] gives conflicts of cand.
                # # We want no True in conflict_matrix[cand, selected] except possibly at sel_idx.
                
                # conflicts_with_selected = conflict_matrix[cand, selected]
                # # Remove the entry corresponding to sel_idx
                # # Since selected is a list, we need to be careful.
                
                # Let's just do the loop, it's small (k~140).
                is_compatible = True
                for j, s in enumerate(selected):
                    if j == i:
                        continue
                    if conflict_matrix[cand, s]:
                        is_compatible = False
                        break
                
                if is_compatible:
                    # Swap
                    selected[i] = cand
                    selected_set.discard(sel_idx)
                    selected_set.add(cand)
                    unselected.remove(sel_idx)
                    unselected.append(cand)
                    improved = True
                    break # Restart the outer loop? Or continue?
                    # Breaking inner loop allows us to try next selected point in next outer iteration?
                    # Or just break both to restart?
                    # Let's break inner, and continue outer.
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_selected = selected[:]
        
        return best_selected

    # Generate pool for w=4
    pool = generate_pool(4)
    n = len(pool)
    
    # Precompute dot products and conflict matrix
    D = pool @ pool.T
    max_dot = 2
    conflict_matrix = D > max_dot
    
    best_result = None
    best_k = 0
    
    np.random.seed(42)
    
    # Run multiple restarts
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        order = np.random.permutation(n)
        selected = greedy_init(pool, conflict_matrix, order)
        
        # Local search
        # Allocate some time for local search per restart
        # If we have 580s total, and 50 restarts, that's ~11s per restart.
        # Greedy is fast (<0.1s). So ~10s for local search.
        ls_time_limit = time.time() + 8 
        selected = local_search(pool, conflict_matrix, selected, ls_time_limit, max_iters=2000)
        
        if len(selected) > best_k:
            best_k = len(selected)
            best_result = pool[selected]
            
            # Print progress? No, keep it clean.
            # print(f"Restart {r}: k={best_k}")

    # Fallback: Weight 2 vectors (always valid, k=220)
    if best_result is None or best_k < 220:
        points2 = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        points2.append(v)
        best_result = np.array(points2, dtype=np.int64)
        best_k = len(best_result)
        
    best_result = np.around(best_result).astype(np.int64)
    return best_result

# EVOLVE-BLOCK-END
