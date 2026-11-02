import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T[a, b, c] is 1 if (a, b, c) corresponds to (i, j, k)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # We store the non-zero indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    target_indices = np.array(target_indices)
    
    # Precompute the full target tensor for quick error calculation?
    # Size 9x15x15 = 2025. Very small.
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for a, b, c in target_indices:
        T[a, b, c] = 1.0
        
    start_time = time.time()
    deadline = 120.0 # Keep 30s margin
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
                
    best_solution = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook
    
    # Try to find lower rank
    # Heuristic lower bound: maybe 20?
    # We'll try a range of ranks
    ranks_to_try = [20, 22, 25, 28, 30, 35, 40, 45]
    
    # Filter out ranks >= best_rank
    ranks_to_try = [r for r in ranks_to_try if r < best_rank]
    
    # If schoolbook is 45, we try 20..40.
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try multiple restarts for this rank
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W
            # Use small random integers or half-integers to help rounding later?
            # Or just random floats.
            U = np.random.uniform(-1, 1, (nm, R))
            V = np.random.uniform(-1, 1, (mp, R))
            W = np.random.uniform(-1, 1, (pn, R))
            
            # Flatten parameters
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def error_func(params):
                U_flat = params[:nm*R]
                V_flat = params[nm*R:nm*R + mp*R]
                W_flat = params[nm*R + mp*R:]
                
                U_ = U_flat.reshape(nm, R)
                V_ = V_flat.reshape(mp, R)
                W_ = W_flat.reshape(pn, R)
                
                # Compute M[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Efficient way:
                # M = np.einsum('ar,br,cr->abc', U_, V_, W_)
                M = np.einsum('ar,br,cr->abc', U_, V_, W_, optimize=True)
                
                # Error
                err = np.sum((M - T)**2)
                return err
            
            # Use L-BFGS-B
            from scipy.optimize import minimize
            try:
                res = minimize(error_func, params, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            except Exception:
                continue
                
            if res.fun < 1e-9:
                # Check if it's close enough to round
                U_opt = res.x[:nm*R].reshape(nm, R)
                V_opt = res.x[nm*R:nm*R + mp*R].reshape(mp, R)
                W_opt = res.x[nm*R + mp*R:].reshape(pn, R)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                # Verify exactness
                M_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded, optimize=True)
                if np.allclose(M_rounded, T, atol=1e-9):
                    best_rank = R
                    best_solution = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    break # Move to next lower rank? No, we are iterating ranks ascending?
                    # Wait, ranks_to_try is ascending. If we find 20, we should stop?
                    # No, we want the MINIMAL rank.
                    # So if we find 20, we can return immediately?
                    # Yes, if we are trying 20, 22, 25... and 20 works, it's minimal in this set.
                    # But maybe 18 works?
                    # We should try lower ranks first.
                    
        if best_rank == R:
            # Found a solution for this rank. Since we are iterating ascending, this is the best so far.
            # We can break if we want to save time, but maybe a lower rank is possible?
            # No, we are iterating 20, 22, 25...
            # If 20 worked, we should have found it in the first block.
            # So if we are at R=25 and found a solution, it means 20 and 22 failed.
            # So 25 is the best found.
            # We can break to save time for polish?
            # Or continue to see if 25 was a fluke and 22 works with more restarts?
            # Given time limits, returning the first found minimal rank is safe.
            break
            
    return best_solution
