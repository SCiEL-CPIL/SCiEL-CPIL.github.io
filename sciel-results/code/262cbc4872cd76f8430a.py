import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 for all i, j, k
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Schoolbook solution (Rank 36)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # Try to find rank 32 solution
    target_rank = 32
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for overhead
    
    best_solution = None
    
    # Residual function
    def residuals(params, U_shape, V_shape, W_shape, n, m, p, rank):
        U = params[:U_shape[0]*rank].reshape(U_shape)
        V = params[U_shape[0]*rank : U_shape[0]*rank + V_shape[0]*rank].reshape(V_shape)
        W = params[U_shape[0]*rank + V_shape[0]*rank :].reshape(W_shape)
        
        # Compute T_reconstructed[i, j, k]
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[i, j, k] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
        return (res - 1.0).flatten()

    U_shape = (n * m, target_rank)
    V_shape = (m * p, target_rank)
    W_shape = (p * n, target_rank)
    total_vars = U_shape[0]*target_rank + V_shape[0]*target_rank + W_shape[0]*target_rank
    
    # Try a few random initializations
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        np.random.seed(42 + r)
        x0 = np.random.randn(total_vars) * 0.5
        
        try:
            result = least_squares(residuals, x0, args=(U_shape, V_shape, W_shape, n, m, p, target_rank), 
                                   method='lm', max_nfev=500)
            
            if result.success or result.cost < 1e-10:
                # Extract U, V, W
                U_opt = result.x[:U_shape[0]*target_rank].reshape(U_shape)
                V_opt = result.x[U_shape[0]*target_rank : U_shape[0]*target_rank + V_shape[0]*target_rank].reshape(V_shape)
                W_opt = result.x[U_shape[0]*target_rank + V_shape[0]*target_rank :].reshape(W_shape)
                
                # Snap to half-integers
                U_snap = np.round(U_opt * 2) / 2
                V_snap = np.round(V_opt * 2) / 2
                W_snap = np.round(W_opt * 2) / 2
                
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_snap[i*m+j] * V_snap[j*p+k] * W_snap[k*n+i])
                            if not np.isclose(val, 1.0, atol=1e-9):
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    best_solution = (U_snap, V_snap, W_snap)
                    break # Found a valid solution
            
        except Exception:
            pass
            
    if best_solution:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
