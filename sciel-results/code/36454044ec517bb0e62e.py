import numpy as np
import time

def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))

def _local_search_fast(a, b, p, deadline, best_cost, best_p):
    n = len(p)
    # Copy p to work on
    p = p.copy()
    # Initialize B_p = b[p, p]
    B_p = b[np.ix_(p, p)].copy()
    cur = _qap_cost(a, b, p)
    
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
        
    # Precompute a rows/cols as views for speed?
    # a is int64
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                # Compute delta for swapping i and j
                # pi = p[i], pj = p[j]
                # We use B_p which is b[p, p]
                
                # s1 = a[i, :] @ (B_p[j, :] - B_p[i, :])
                # s2 = a[j, :] @ (B_p[i, :] - B_p[j, :])
                # s3 = a[:, i] @ (B_p[:, j] - B_p[:, i])
                # s4 = a[:, j] @ (B_p[:, i] - B_p[:, j])
                
                # Note: The formula derived earlier:
                # delta = sum_k (a[i,k]-a[j,k]) (B_p[k,j] - B_p[k,i]) ... wait.
                # Let's re-verify the indices for B_p.
                # B_p[k, l] = b[p[k], p[l]].
                # Term 5: sum_{k!=i,j} a[i,k] (b[pj, p[k]] - b[pi, p[k]])
                # b[pj, p[k]] = B_p[j, k] (since p[j]=pj)
                # b[pi, p[k]] = B_p[i, k]
                # So term is sum_k a[i,k] (B_p[j, k] - B_p[i, k])
                # This is a[i, :] @ (B_p[j, :] - B_p[i, :])
                # BUT we must exclude k=i and k=j.
                # The dot product includes k=i and k=j.
                # Let's calculate the full dot product and subtract the i, j terms.
                
                diff_row = B_p[j, :] - B_p[i, :]
                s1_full = int(a[i, :] @ diff_row)
                # Subtract k=i term: a[i,i] * (B_p[j,i] - B_p[i,i])
                # Subtract k=j term: a[i,j] * (B_p[j,j] - B_p[i,j])
                s1 = s1_full - int(a[i, i] * (B_p[j, i] - B_p[i, i])) - int(a[i, j] * (B_p[j, j] - B_p[i, j]))
                
                diff_row_rev = B_p[i, :] - B_p[j, :]
                s2_full = int(a[j, :] @ diff_row_rev)
                s2 = s2_full - int(a[j, j] * (B_p[i, j] - B_p[j, j])) - int(a[j, i] * (B_p[i, i] - B_p[j, i]))
                
                diff_col = B_p[:, j] - B_p[:, i]
                s3_full = int(a[:, i] @ diff_col)
                s3 = s3_full - int(a[i, i] * (B_p[j, i] - B_p[i, i])) - int(a[j, i] * (B_p[j, j] - B_p[i, j]))
                
                diff_col_rev = B_p[:, i] - B_p[:, j]
                s4_full = int(a[:, j] @ diff_col_rev)
                s4 = s4_full - int(a[i, j] * (B_p[j, i] - B_p[i, i])) - int(a[j, j] * (B_p[j, j] - B_p[i, j]))
                
                # Direct terms for i, j interaction
                # Original: a[i,j]b[pi,pj] + a[j,i]b[pj,pi] -> a[i,j]B_p[i,j] + a[j,i]B_p[j,i]
                # New: a[i,j]b[pj,pi] + a[j,i]b[pi,pj] -> a[i,j]B_p[j,i] + a[j,i]B_p[i,j]
                delta_direct = int(a[i, j] * (B_p[j, i] - B_p[i, j]) + a[j, i] * (B_p[i, j] - B_p[j, i]))
                
                delta = s1 + s2 + s3 + s4 + delta_direct
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    cur += delta
                    improved = True
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    
                    # Update B_p
                    # Swap rows i, j and cols i, j
                    B_p[i, :], B_p[j, :] = B_p[j, :].copy(), B_p[i, :].copy()
                    B_p[:, i], B_p[:, j] = B_p[:, j].copy(), B_p[:, i].copy()
                    # Fix the diagonal/intersection points which were swapped twice or incorrectly
                    # B_p[i, i] should be b[p[i], p[i]] = b[pj, pj] = old B_p[j, j]
                    # B_p[j, j] should be b[pj, pi] ... wait.
                    # After swap:
                    # p[i] = old pj, p[j] = old pi.
                    # B_p[i, i] = b[old pj, old pj] = old B_p[j, j]
                    # B_p[j, j] = b[old pi, old pi] = old B_p[i, i]
                    # B_p[i, j] = b[old pj, old pi] = old B_p[j, i]
                    # B_p[j, i] = b[old pi, old pj] = old B_p[i, j]
                    
                    # The row/col swap logic above:
                    # 1. Swap rows i, j.
                    #    New Row i = Old Row j.
                    #    New Row j = Old Row i.
                    # 2. Swap cols i, j.
                    #    New Col i = Old Col j (from step 1).
                    #    New Col j = Old Col i (from step 1).
                    
                    # Let's trace B_p[i, i]:
                    # Step 1: B_p[i, i] becomes Old B_p[j, i].
                    # Step 2: Col i is swapped. B_p[i, i] is in Col i.
                    #         It moves to B_p[i, j]? No, we swap columns.
                    #         Element at (i, i) moves to (i, j) if we swap cols i, j?
                    #         Yes, (row, col) -> (row, new_col).
                    #         So (i, i) -> (i, j).
                    #         And (i, j) -> (i, i).
                    
                    # This is getting confusing. Easier to just recompute the 4 affected cells.
                    # Or just recompute the whole B_p? O(n^2) = 10,000 ops.
                    # Given n=100, recomputing B_p is 10k ops.
                    # Delta calc is 4 * 100 = 400 ops.
                    # If we recompute B_p, we spend 10k ops per swap.
                    # Pass has ~100 swaps (if many improvements). 100 * 10k = 1M ops.
                    # This is acceptable.
                    
                    B_p = b[np.ix_(p, p)]
                    
                    break # Restart inner loop
        if not improved:
            break
            
    return best_cost, best_p

def _anneal(a, b, p, deadline, best_cost, best_p, rng):
    n = len(p)
    p = p.copy()
    B_p = b[np.ix_(p, p)].copy()
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
        
    t = 100.0
    alpha = 0.99
    t_end = 1.0
    
    while time.time() < deadline and t > t_end:
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
            
        # Compute delta
        diff_row = B_p[j, :] - B_p[i, :]
        s1_full = int(a[i, :] @ diff_row)
        s1 = s1_full - int(a[i, i] * (B_p[j, i] - B_p[i, i])) - int(a[i, j] * (B_p[j, j] - B_p[i, j]))
        
        diff_row_rev = B_p[i, :] - B_p[j, :]
        s2_full = int(a[j, :] @ diff_row_rev)
        s2 = s2_full - int(a[j, j] * (B_p[i, j] - B_p[j, j])) - int(a[j, i] * (B_p[i, i] - B_p[j, i]))
        
        diff_col = B_p[:, j] - B_p[:, i]
        s3_full = int(a[:, i] @ diff_col)
        s3 = s3_full - int(a[i, i] * (B_p[j, i] - B_p[i, i])) - int(a[j, i] * (B_p[j, j] - B_p[i, j]))
        
        diff_col_rev = B_p[:, i] - B_p[:, j]
        s4_full = int(a[:, j] @ diff_col_rev)
        s4 = s4_full - int(a[i, j] * (B_p[j, i] - B_p[i, i])) - int(a[j, j] * (B_p[j, j] - B_p[i, j]))
        
        delta_direct = int(a[i, j] * (B_p[j, i] - B_p[i, j]) + a[j, i] * (B_p[i, j] - B_p[j, i]))
        delta = s1 + s2 + s3 + s4 + delta_direct
        
        if delta < 0 or rng.random() < np.exp(-delta / max(t, 1e-10)):
            p[i], p[j] = p[j], p[i]
            cur += delta
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
            B_p = b[np.ix_(p, p)]
            
        t *= alpha
        
    # Final local search
    best_cost, best_p = _local_search_fast(a, b, best_p, deadline, best_cost, best_p)
    return best_cost, best_p

def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 115.0
    
    rng = np.random.default_rng(12345)
    
    best_cost = np.iinfo(np.int64).max
    best_p = None
    
    # Multiple restarts
    # Each restart: init -> local search -> anneal
    # Time per restart: ~10-20s?
    # Let's try to fit 5-10 restarts.
    
    while time.time() < deadline - 10.0:
        # Strategy 1: Random
        p = rng.permutation(n).astype(np.int64)
        # Strategy 2: Greedy (if time permits, but random is faster)
        
        # Run local search
        c, p_best = _local_search_fast(a, b, p, deadline - 5.0, best_cost, best_p)
        if c < best_cost:
            best_cost = c
            best_p = p_best
        
        # Run anneal if time
        if time.time() < deadline - 15.0:
            c, p_best = _anneal(a, b, best_p, deadline - 5.0, best_cost, best_p, rng)
            if c < best_cost:
                best_cost = c
                best_p = p_best
        
    # Final polish
    if time.time() < deadline - 2.0:
        c, p_best = _local_search_fast(a, b, best_p, deadline, best_cost, best_p)
        if c < best_cost:
            best_cost = c
            best_p = p_best
            
    return best_p.tolist()
