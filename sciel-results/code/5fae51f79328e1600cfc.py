import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [5, 5, 5]
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # 1. Initialize with Schoolbook Decomposition (Rank 125)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_rank = rank

    # Helper: Compute Tensor T from U, V, W
    def compute_tensor(U_mat, V_mat, W_mat):
        U_reshaped = U_mat.reshape(n, m, -1)
        V_reshaped = V_mat.reshape(m, p, -1)
        W_reshaped = W_mat.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        return T

    # Helper: Error function - sum of squared errors
    def error_func(x, r):
        U_curr = x[:n*m*r].reshape(n*m, r)
        V_curr = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W_curr = x[n*m*r + m*p*r : n*m*r + m*p*r + p*n*r].reshape(p*n, r)
        
        T = compute_tensor(U_curr, V_curr, W_curr)
        error = np.sum((T - 1.0)**2)
        return error

    # Helper: Check if a decomposition is valid
    def is_valid(U_mat, V_mat, W_mat, r):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            return False
        T = compute_tensor(U_mat, V_mat, W_mat)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-8)

    # Strategy: Reduce rank by zeroing out columns and re-optimizing
    # Start from schoolbook, try to remove one column at a time
    
    current_U = best_U.copy()
    current_V = best_V.copy()
    current_W = best_W.copy()
    current_rank = best_rank
    
    # Try to reduce rank from current_rank down
    # For each target rank, try multiple random column removals
    for target_rank in range(current_rank - 1, 44, -1):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Only try a few attempts per rank to stay within time budget
        max_attempts = 3
        found = False
        
        for attempt in range(max_attempts):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Randomly select columns to zero out
            # We want to go from current_rank to target_rank
            cols_to_zero = np.random.choice(current_rank, size=current_rank - target_rank, replace=False)
            
            # Create masked version
            U_masked = current_U.copy()
            V_masked = current_V.copy()
            W_masked = current_W.copy()
            U_masked[:, cols_to_zero] = 0.0
            V_masked[:, cols_to_zero] = 0.0
            W_masked[:, cols_to_zero] = 0.0
            
            # Now optimize the remaining columns
            # Extract active columns
            active_cols = np.setdiff1d(np.arange(current_rank), cols_to_zero)
            U_active = U_masked[:, active_cols]
            V_active = V_masked[:, active_cols]
            W_active = W_masked[:, active_cols]
            
            r = target_rank
            x0 = np.concatenate([U_active.flatten(), V_active.flatten(), W_active.flatten()])
            
            # Optimize
            try:
                res = minimize(lambda x: error_func(x, r), x0, method='BFGS', 
                              options={'maxiter': 50, 'disp': False, 'gtol': 1e-6})
                
                if res.fun < 1e-6:
                    U_sol = res.x[:n*m*r].reshape(n*m, r)
                    V_sol = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_sol = res.x[n*m*r + m*p*r : n*m*r + m*p*r + p*n*r].reshape(p*n, r)
                    
                    if is_valid(U_sol, V_sol, W_sol, r):
                        best_U = U_sol
                        best_V = V_sol
                        best_W = W_sol
                        best_rank = r
                        current_U = U_sol
                        current_V = V_sol
                        current_W = W_sol
                        current_rank = r
                        found = True
                        break
                        
            except Exception:
                continue
        
        if found:
            continue
        else:
            # If we can't reduce further from this starting point, try different randomizations
            # But to save time, we'll stop if we've spent too much time
            if time.time() - start_time > TIME_LIMIT - 20.0:
                break
    
    # Final polish: try to verify and return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
