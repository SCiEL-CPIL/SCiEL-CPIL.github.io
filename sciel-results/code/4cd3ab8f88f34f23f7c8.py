import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    best_rank = n * m * p  # 36
    best_result = None
    
    # We try ranks from a lower bound up to the schoolbook rank.
    # Lower bound is max(nm, mp, pn) = 18.
    # We prioritize finding a rank significantly lower than 36.
    # We will search from 18 upwards, but stop if we find a valid one.
    # However, lower ranks are harder. 
    # It is often better to start from a rank we think is achievable, e.g., 20, 21, 22.
    # Let's try 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35.
    
    # To save time, we will try a few random initializations for each rank.
    # If we find a solution, we return it immediately.
    
    ranks_to_try = list(range(20, 36)) + [18, 19]
    # Sort by likelihood? 20-25 is a good guess.
    # Let's just iterate.
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # If we already have a better result, skip higher ranks?
        # No, we want the MINIMAL rank. So we should try lower ranks first.
        # But if rank 20 fails, we try 21, etc.
        # If we find rank 20, we break and return 20.
        # If we don't find any < 36, we return 36.
        
        # Number of trials per rank.
        # Higher ranks are easier to find.
        # Lower ranks are harder.
        trials = 20 if rank < 25 else 10
        
        for trial in range(trials):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(1000 + rank * 100 + trial)
            
            # Random initialization
            # Scale is important. We use small values to start.
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # To help convergence, we can project onto a manifold where 
            # one of the matrices is normalized? 
            # For now, just random.
            
            def residuals(params):
                U_r = params[:nm*rank].reshape(nm, rank)
                V_r = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W_r = params[nm*rank+mp*rank:].reshape(pn, rank)
                
                # Compute T_approx
                # T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Using einsum
                # 'ar,br,cr->abc'
                T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
                
                diff = T_approx - T
                return diff.ravel()
            
            x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            
            try:
                from scipy.optimize import least_squares
                # Use 'trf' for large problems, but this is small.
                # max_nfev: limit function evaluations to prevent hanging.
                # 1296 residuals.
                result = least_squares(residuals, x0, method='trf', max_nfev=500)
                
                if result.success or np.linalg.norm(result.fun) < 1e-5:
                    U_r = result.x[:nm*rank].reshape(nm, rank)
                    V_r = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_r = result.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    # Verify
                    T_approx = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
                    if np.max(np.abs(T_approx - T)) < 1e-5:
                        best_rank = rank
                        best_result = (U_r, V_r, W_r)
                        break  # Found valid decomposition for this rank, try next lower rank?
                        # No, we are iterating from 20 upwards. 
                        # If we find rank 20, we break the inner loop.
                        # Then we should break the outer loop? 
                        # Yes, because we want the MINIMAL rank. 
                        # Wait, I am iterating 20, 21, 22...
                        # So if I find 20, I should return it.
                        # If I don't find 20, I try 21.
                        # So breaking the inner loop is correct.
                        # But I need to break the outer loop too to return the best (lowest) found.
                        # Since I'm going up, the first one found is the lowest.
            
            except Exception:
                pass
        
        if best_result is not None:
            break
    
    # If no better found, return schoolbook
    if best_result is None:
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U_r, V_r, W_r = best_result
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
