import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Precompute Target Tensor
    TARGET_TENSOR = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                TARGET_TENSOR[u, v, w] = 1.0

    # Precompute Schoolbook Solution (Rank 48)
    R_school = n * m * p
    U_school = np.zeros((size_u, R_school))
    V_school = np.zeros((size_v, R_school))
    W_school = np.zeros((size_w, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    def tensor_contraction(U, V, W):
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        R = len(params) // (size_u + size_v + size_w)
        if R == 0: return 1e10
        
        U = params[:size_u * R].reshape(size_u, R)
        V = params[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
        W = params[size_u * R + size_v * R :].reshape(size_w, R)
        
        T = tensor_contraction(U, V, W)
        
        # Use sum of squares
        return np.sum((T - TARGET_TENSOR)**2)

    best_solution = None
    best_rank = R_school
    best_params = None
    
    # Fallback
    fallback = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # Search for lower ranks
    # We iterate downwards.
    # For each rank, we try multiple initializations.
    
    ranks_to_try = list(range(47, 44, -1)) # Down to 45
    
    for R in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        # Determine number of restarts
        time_left = deadline - time.time()
        if time_left > 60:
            n_restarts = 15
        elif time_left > 30:
            n_restarts = 8
        else:
            n_restarts = 3
            
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + R * 100 + restart)
            
            # Initialization Strategy: Perturbed Schoolbook
            # Take the first R columns of the schoolbook solution and add small noise
            # Or take a random subset.
            
            if restart % 2 == 0:
                # Random Gaussian
                params = np.random.randn(size_u * R + size_v * R + size_w * R) * 0.1
            else:
                # Random Subset of Schoolbook Columns
                indices = np.random.choice(R_school, R, replace=False)
                U_init = U_school[:, indices]
                V_init = V_school[:, indices]
                W_init = W_school[:, indices]
                # Add small noise to escape local minima
                U_init += np.random.randn(size_u, R) * 0.01
                V_init += np.random.randn(size_v, R) * 0.01
                W_init += np.random.randn(size_w, R) * 0.01
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
            try:
                res = minimize(objective, params, method='L-BFGS-B', 
                               options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-6: # Close to zero error
                    R_opt = R
                    U_opt = res.x[:size_u * R_opt].reshape(size_u, R_opt)
                    V_opt = res.x[size_u * R_opt : size_u * R_opt + size_v * R_opt].reshape(size_v, R_opt)
                    W_opt = res.x[size_u * R_opt + size_v * R_opt :].reshape(size_w, R_opt)
                    
                    # Snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    # Verify snapped solution
                    T_snap = tensor_contraction(U_snap, V_snap, W_snap)
                    if np.allclose(T_snap, TARGET_TENSOR, atol=1e-9):
                        best_rank = R
                        best_solution = {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        break 
                        
            except Exception:
                pass

    if best_solution:
        return best_solution
    else:
        return fallback

# EVOLVE-BLOCK-END
