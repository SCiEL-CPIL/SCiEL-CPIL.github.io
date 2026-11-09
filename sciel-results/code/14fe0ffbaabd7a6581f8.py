"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    n, m, p = DIMENSIONS
    # Dimensions of U, V, W
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Precompute index mappings for vectorization
    # i, j, k -> index in U, V, W
    # U_idx[i, j] = i * m + j
    # V_idx[j, k] = j * p + k
    # W_idx[k, i] = k * n + i
    
    # We want to find U, V, W such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # Let's try ranks from lower bound up
    lower_bound = max(dim_U, dim_V, dim_W)
    
    # Fallback: Schoolbook
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_U, schoolbook_rank))
    V_sb = np.zeros((dim_V, schoolbook_rank))
    W_sb = np.zeros((dim_W, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    deadline = time.time() + 140 # 10s margin
    
    # Try ranks
    for rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline:
            break
            
        # Try a few seeds
        found = False
        for seed in range(10):
            if time.time() > deadline:
                break
            rng = np.random.RandomState(seed)
            
            # Initialize
            U = rng.randn(dim_U, rank) * 0.1
            V = rng.randn(dim_V, rank) * 0.1
            W = rng.randn(dim_W, rank) * 0.1
            
            # Gauss-Newton Optimization
            # Objective: sum (sum_r U_ijk V_jkr W_kir - 1)^2
            # We can use scipy.optimize.least_squares for robustness, but it might be slow.
            # Let's implement a simple Gauss-Newton or use minimize with BFGS.
            
            # Flatten
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals(x):
                U_x = x[:dim_U*rank].reshape(dim_U, rank)
                V_x = x[dim_U*rank:dim_U*rank+dim_V*rank].reshape(dim_V, rank)
                W_x = x[dim_U*rank+dim_V*rank:].reshape(dim_W, rank)
                
                res = np.zeros(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_x[i*m+j] * V_x[j*p+k] * W_x[k*n+i])
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            # Try to solve
            try:
                from scipy.optimize import least_squares
                # Use 'trf' or 'dogbox'
                # Provide initial guess
                # We can speed this up by using a vectorized residual calculation
                # But for small dims, the loop is fine.
                
                # To improve convergence, let's use a smooth loss and minimize
                # Actually, least_squares is good for this.
                # However, it's iterative.
                
                # Alternative: Directly solve linear system if we fix two matrices?
                # No, let's stick to least_squares.
                
                # We need to be careful with time.
                # Let's limit iterations.
                
                # A faster approach:
                # Since we only have 48 constraints, we can try to satisfy them directly.
                # But the non-linearity makes it hard.
                
                # Let's try a few iterations of a custom gradient step with line search.
                # Or just use least_squares with a tight tolerance.
                
                # To save time, we only run least_squares for a few seconds per seed.
                # But least_squares doesn't have a time limit, only max_iter.
                # We can wrap it in a thread? No, keep it simple.
                
                # Let's try a heuristic:
                # If rank is close to lower bound, it might be hard.
                # Let's try rank = lower_bound first.
                
                # Run least_squares
                # max_iter=1000 is usually enough for this size.
                sol = least_squares(residuals, x0, method='trf', max_iter=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                if sol.success and sol.cost < 1e-10:
                    U_sol = sol.x[:dim_U*rank].reshape(dim_U, rank)
                    V_sol = sol.x[dim_U*rank:dim_U*rank+dim_V*rank].reshape(dim_V, rank)
                    W_sol = sol.x[dim_U*rank+dim_V*rank:].reshape(dim_W, rank)
                    
                    # Verify
                    max_res = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_sol[i*m+j] * V_sol[j*p+k] * W_sol[k*n+i])
                                max_res = max(max_res, abs(val - 1.0))
                    
                    if max_res < 1e-9:
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        best_rank = rank
                        found = True
                        break # Found for this rank, try lower? No, we iterate rank up. So break seeds, go to next rank?
                        # Wait, we want MINIMAL rank. So we start from lower_bound.
                        # If found, we can return immediately?
                        # The loop goes from lower_bound to schoolbook.
                        # So if found at rank R, that is the minimal rank (assuming lower_bound is correct lower bound).
                        # So we can return.
                        
            except Exception as e:
                pass
        
        if found:
            break
    
    # Convert to list
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
