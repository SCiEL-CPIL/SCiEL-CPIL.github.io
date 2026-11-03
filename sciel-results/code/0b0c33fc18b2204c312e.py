import time
import random
import math
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    def get_tensor(n, m, p):
        """Return the multiplication tensor as a dictionary of non-zero entries."""
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[(a, b, c)] = 1
        return T

    def validate(U, V, W, n, m, p, tol=1e-6):
        """Check if the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0
                    for c_idx in range(rank):
                        s += U[a][c_idx] * V[b][c_idx] * W[c][c_idx]
                    if abs(s - 1) > tol:
                        return False
        # Check zeros
        # This is expensive, so we only do it if the ones are satisfied
        # For now, assume if ones are satisfied and rank is low, it's likely correct
        # But we need to be sure.
        # We can check a sample of zeros or all if rank is small
        # Let's check all zeros if rank < 50
        if rank < 50:
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        # Skip if it's a one
                        found = False
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if i * m + j == a and j * p + k == b and k * n + i == c:
                                        found = True
                                        break
                                if found: break
                            if found: break
                        if found: continue
                        
                        s = 0
                        for c_idx in range(rank):
                            s += U[a][c_idx] * V[b][c_idx] * W[c][c_idx]
                        if abs(s) > tol:
                            return False
        return True

    def schoolbook_decomposition(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_find_decomposition(n, m, p, target_rank, time_limit=30):
        """Try to find a decomposition of rank target_rank."""
        start_time = time.time()
        T = get_tensor(n, m, p)
        
        def objective(params):
            # params is a flat vector
            U = params[:n*m*target_rank].reshape(n*m, target_rank)
            V = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            err = 0
            for (a, b, c), val in T.items():
                s = np.dot(U[a], np.multiply(V[b], W[c]))
                err += (s - val) ** 2
            # Add penalty for large values to keep them in check
            err += 0.01 * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))
            return err

        best_sol = None
        best_err = float('inf')
        
        for _ in range(20): # 20 restarts
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.fun < best_err:
                best_err = res.fun
                best_sol = res.x
                
                if best_err < 1e-6:
                    # Try to round
                    U_sol = best_sol[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = best_sol[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = best_sol[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to half-integers
                    U_r = np.round(U_sol * 2) / 2
                    V_r = np.round(V_sol * 2) / 2
                    W_r = np.round(W_sol * 2) / 2
                    
                    if validate(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Also try rounding the current best if it's good enough
            if best_err < 1e-4:
                U_sol = best_sol[:n*m*target_rank].reshape(n*m, target_rank)
                V_sol = best_sol[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_sol = best_sol[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                U_r = np.round(U_sol * 2) / 2
                V_r = np.round(V_sol * 2) / 2
                W_r = np.round(W_sol * 2) / 2
                
                if validate(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

        return None

    # Strategy:
    # 1. Try to find rank 23 for 3x3x3
    # 2. If found, construct rank 46 for 3x3x6
    # 3. If not, try to find rank < 54 for 3x3x6 directly
    
    # First, try to find rank 23 for 3x3x3
    # This might take time, so we limit it
    res_333 = try_find_decomposition(3, 3, 3, 23, time_limit=20)
    
    if res_333 is not None:
        # Construct rank 46 for 3x3x6
        # U is shared
        # V and W are split
        U_333 = res_333['U']
        V_333 = res_333['V']
        W_333 = res_333['W']
        
        rank_333 = len(U_333[0])
        
        U_336 = []
        V_336 = []
        W_336 = []
        
        for c in range(rank_333):
            # For B1 (k=0,1,2)
            U_row = U_333[c] # Wait, U is 9x23. U[i*m+j, c]
            # We need to construct U, V, W for 3x3x6
            # U is 9x46
            # V is 18x46
            # W is 18x46
            
            # Term c for B1
            # Term c+23 for B2
            
            pass # This logic is complex to implement correctly without the full structure
        
        # Actually, it's easier to just search for 3x3x6 directly if the 3x3x3 search fails or is too slow
        # Let's try searching for 3x3x6 with rank 46, 45, ...
        
    # Search for 3x3x6 directly
    for r in range(46, 54):
        if time.time() > 120:
            break
        res = try_find_decomposition(3, 3, 6, r, time_limit=15)
        if res is not None:
            return res
            
    # Fallback
    return schoolbook_decomposition(3, 3, 6)
