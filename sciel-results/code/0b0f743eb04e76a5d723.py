import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Checks if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    # Reshape for easier indexing
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R) # W index is k*n+i, so W_3d[k, i, r]
    
    # Compute the tensor T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Using einsum
    # U[i, j, r], V[j, k, r], W[k, i, r]
    # Result should be 1 for all i, j, k
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Check if T is all ones
    if not np.allclose(T, np.ones((n, m, p)), atol=tol):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # 1. Construct Schoolbook
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    # Verify schoolbook (should be True)
    # if not check_decomposition(U_sb, V_sb, W_sb, n, m, p):
    #     return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # 2. Try to find a lower rank decomposition
    # We will try to optimize a rank R < 54 decomposition.
    # Start with R = 53.
    
    deadline = time.time() + 100 # Leave 20s margin
    
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_schoolbook
    
    # Try ranks from 53 down to maybe 50 if time permits
    for target_rank in range(rank_schoolbook - 1, rank_schoolbook - 5, -1):
        if time.time() > deadline:
            break
            
        # Initialize with schoolbook, zero out some columns
        # We take the first 'target_rank' columns of the schoolbook?
        # No, that would leave some (i,j,k) equations unsatisfied.
        # Better: Start with full schoolbook, set (54-target_rank) columns to 0,
        # and optimize the rest.
        
        U_opt = U_sb[:, :target_rank].copy()
        V_opt = V_sb[:, :target_rank].copy()
        W_opt = W_sb[:, :target_rank].copy()
        
        # We dropped the last (54 - target_rank) columns.
        # This creates errors in the equations corresponding to those columns.
        # We want to adjust U_opt, V_opt, W_opt to fix these errors.
        
        def objective(params):
            U_flat = params[:n*m*target_rank]
            V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = params[n*m*target_rank + m*p*target_rank :]
            
            U_curr = U_flat.reshape(n, m, target_rank)
            V_curr = V_flat.reshape(m, p, target_rank)
            W_curr = W_flat.reshape(p, n, target_rank)
            
            T = np.einsum('ijr,jkr,kir->ijk', U_curr, V_curr, W_curr)
            error = np.sum((T - 1.0)**2)
            return error
        
        initial_params = np.concatenate([
            U_opt.flatten(),
            V_opt.flatten(),
            W_opt.flatten()
        ])
        
        # Run optimization
        try:
            res = minimize(objective, initial_params, method='L-BFGS-B', 
                           options={'maxiter': 50, 'maxfun': 500, 'disp': False})
            
            if res.fun < 1e-6:
                # Extract solution
                U_sol = res.x[:n*m*target_rank].reshape(n, m, target_rank)
                V_sol = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m, p, target_rank)
                W_sol = res.x[n*m*target_rank + m*p*target_rank :].reshape(p, n, target_rank)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                # Check exactness
                if check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p, tol=1e-9):
                    best_rank = target_rank
                    best_result = {
                        "U": U_rounded.reshape(n*m, target_rank).tolist(),
                        "V": V_rounded.reshape(m*p, target_rank).tolist(),
                        "W": W_rounded.reshape(p*n, target_rank).tolist()
                    }
                    break # Found a better one, stop searching lower ranks for now
        except Exception:
            pass

    return best_result
