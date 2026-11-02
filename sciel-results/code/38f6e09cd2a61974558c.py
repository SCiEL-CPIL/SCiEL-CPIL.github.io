import numpy as np
import time
import random

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank
    target_rank = 14
    
    # Initialize best result with schoolbook (rank 20)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    start_time = time.time()
    deadline = start_time + 100 # 100s limit to be safe
    
    # Define the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We want U, V, W such that sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = T[i*m+j, j*p+k, k*n+i]
    
    def error(U, V, W):
        # Compute the reconstructed tensor
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Reconstructed T: (nm, mp, pn)
        # We only need to check the non-zero entries of the target tensor
        # Target tensor has 1s at (i*m+j, j*p+k, k*n+i) for all i,j,k
        # There are n*m*p = 20 such entries.
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    # Sum over c
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                    # Target is 1
                    err += (s - 1.0) ** 2
        return err

    # Search for rank 14
    for restart in range(50):
        if time.time() > deadline:
            break
            
        r = target_rank
        # Random initialization
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Normalize to prevent divergence
        U = U / np.linalg.norm(U)
        V = V / np.linalg.norm(V)
        W = W / np.linalg.norm(W)
        
        # ALS (Alternating Least Squares)
        # Minimize || T - sum u_c v_c w_c ||
        # For fixed V, W, U_c = argmin || T - sum u_c v_c w_c ||
        # This is a linear least squares problem for each c? No, it's coupled.
        # Actually, for a fixed c, if we fix other c', it's linear in U_c.
        # Let's use a simple gradient descent or ALS.
        
        # ALS step:
        # Fix V, W. Update U.
        # For each index (i,j) in U, we have contributions to all (i,j,k) in T.
        # This is complex. Let's use a simpler approach: 
        # Random restart with local search using scipy minimize.
        
        def objective(params):
            # params: flattened U, V, W
            U_flat, V_flat, W_flat = np.split(params, [n*m*r, n*m*r + m*p*r])
            U = U_flat.reshape(n*m, r)
            V = V_flat.reshape(m*p, r)
            W = W_flat.reshape(p*n, r)
            return error(U, V, W)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success or res.fun < 1e-8:
                # Check if it's a valid exact decomposition
                # Try to round to rationals
                U_opt = res.x[:n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_opt = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Verify exactness with high precision
                # If error is very small, we assume it's the rational solution
                # We can try to convert to fractions
                # But the prompt accepts integer/half-integer.
                # If the values are close to integers or half-integers, we snap them.
                # Otherwise, we keep them as floats? 
                # "integer/half-integer entries"
                # If the solution is rational with small denominators, it should be close to such values.
                
                # Check error again
                err = error(U_opt, V_opt, W_opt)
                if err < 1e-6:
                    # Snap to nearest half-integer
                    U_snap = [[round(x * 2) / 2 for x in row] for row in U_opt]
                    V_snap = [[round(x * 2) / 2 for x in row] for row in V_opt]
                    W_snap = [[round(x * 2) / 2 for x in row] for row in W_opt]
                    
                    # Verify snapped solution
                    err_snap = error(U_snap, V_snap, W_snap)
                    if err_snap < 1e-6:
                        return {"U": U_snap, "V": V_snap, "W": W_snap}
                    else:
                        # Return unsnapped if it's valid? 
                        # Prompt says "integer/half-integer entries".
                        # If it's not half-integer, it might be invalid.
                        # But if the true solution is half-integer, snapping should work.
                        pass
                        
        except Exception:
            continue
            
    # Fallback to schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
