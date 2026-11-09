import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook baseline
    rank_sb = n * m * p  # 54
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
                
    # Build the multiplication tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Pre-compute reshaped tensors for faster lstsq
    T_a = T.reshape(nm, mp * pn)
    T_b = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T_c = T.transpose(2, 0, 1).reshape(pn, nm * mp)

    start_time = time.time()
    time_limit = 110  # Leave 40s margin for 150s limit? No, 120s is the hard limit. 10s margin is safer.
    # Actually, the prompt says "finish within 120s". Let's use 110s.
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Lower bound for rank
    # R >= (nm + mp + pn) / 2
    lower_bound = int(np.ceil((nm + mp + pn) / 2.0))
    
    # Ranks to try: From schoolbook-1 down to lower_bound
    # We prioritize lower ranks as they are better scores.
    # But lower ranks are harder.
    # Let's try a mix.
    ranks_to_try = list(range(rank_sb - 1, lower_bound - 1, -1))
    
    # If we find a solution, we return it immediately? 
    # The task asks for "rank-minimal". So we must find the lowest possible rank.
    # We should iterate from lower_bound up to rank_sb - 1?
    # If we iterate from low to high, we can stop at the first success.
    # If we iterate from high to low, we might find a high rank solution quickly but miss a lower one.
    # Since we want the MINIMAL rank, we should search from low to high.
    # However, low rank solutions are rare/hard to find.
    # High rank solutions are easier.
    # Strategy: 
    # 1. Try a few low ranks with many restarts.
    # 2. If none found, try mid ranks.
    # 3. If none found, return schoolbook.
    
    # Let's reorder ranks to try lower ones first, but with a time budget per rank.
    # Or just iterate low to high.
    
    ranks_to_try = list(range(lower_bound, rank_sb))
    
    for rank in ranks_to_try:
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
        
        # Determine restarts based on rank
        # Lower rank -> more restarts
        # Higher rank -> fewer restarts
        if rank < 35:
            max_restarts = 20
        elif rank < 45:
            max_restarts = 10
        else:
            max_restarts = 5
            
        found = False
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5:
                break
            
            np.random.seed(12345 + restart * 999 + rank * 31)
            
            # Initialization
            # Random Gaussian
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Normalize to prevent scale issues
            U /= (np.linalg.norm(U) + 1e-10)
            V /= (np.linalg.norm(V) + 1e-10)
            W /= (np.linalg.norm(W) + 1e-10)
            
            # ALS
            max_iter = 100
            prev_error = float('inf')
            for iteration in range(max_iter):
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Solve for U
                # A_u[r, (b,c)] = V[b, r] * W[c, r]
                A_u = np.einsum('br,cr->rbc', V, W).reshape(rank, mp * pn)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A_u, T_a.T, rcond=None)
                    U = U_new.T
                except:
                    break
                
                # Solve for V
                A_v = np.einsum('ar,cr->rac', U, W).reshape(rank, nm * pn)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A_v, T_b.T, rcond=None)
                    V = V_new.T
                except:
                    break
                
                # Solve for W
                A_w = np.einsum('ar,br->rab', U, V).reshape(rank, nm * mp)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A_w, T_c.T, rcond=None)
                    W = W_new.T
                except:
                    break
                
                # Check error
                if iteration % 5 == 0 or iteration == max_iter - 1:
                    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                    error = np.linalg.norm(T_recon - T)
                    
                    # If error is very small, we have a solution
                    if error < 1e-8:
                        found = True
                        break
                        
                    # Stagnation check
                    if error > prev_error * 1.01 and iteration > 10:
                        break
                    prev_error = error
            
            if found:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                T_recon_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                error_snap = np.linalg.norm(T_recon_snap - T)
                
                if error_snap < 1e-6:
                    best_rank = rank
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    # We found a solution for this rank. 
                    # Since we are iterating from low to high, this is the minimal rank found so far.
                    # We should try to see if a lower rank exists? 
                    # No, we are iterating low to high. So this is the first one found.
                    # We can return immediately? 
                    # Yes, if we are confident this is the global minimum.
                    # But what if rank 25 works and we found rank 26?
                    # We are iterating from lower_bound (26) upwards.
                    # So if we find rank 26, we should return it, UNLESS we are sure 25 is impossible.
                    # The lower bound is 26. So 26 is the absolute minimum.
                    # If we find 26, we return.
                    # If we don't find 26, we try 27, etc.
                    # So returning immediately upon finding a valid solution at the current rank is correct 
                    # IF we are iterating from the theoretical minimum upwards.
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                else:
                    # Snapping failed. Maybe the solution is valid but not half-integer?
                    # The contract says "integer/half-integer entries".
                    # If we can't snap, we might have to discard it or try to polish.
                    # Let's try a local polish around the snapped values?
                    # Or just discard and continue searching.
                    pass
        
        # If we didn't find a solution for this rank, continue to next rank

    # If no lower rank found, return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
