import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Indices:
    # U: (n*m, R) -> index i*m + j
    # V: (m*p, R) -> index j*p + k
    # W: (p*n, R) -> index k*n + i  (Note: prompt says W[k*n+i, r])
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Precompute the list of (u, v, w) indices where T is 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_indices.append((u, v, w))
                
    def tensor_contraction(U, V, W):
        return np.einsum('ur,vr,wr->uvw', U, V, W)

    # Initialize Target Tensor
    TARGET_TENSOR = np.zeros((size_u, size_v, size_w))
    for u, v, w in target_indices:
        TARGET_TENSOR[u, v, w] = 1.0

    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        R = len(params) // (size_u + size_v + size_w)
        if R == 0: return 1e10
        
        U = params[:size_u * R].reshape(size_u, R)
        V = params[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
        W = params[size_u * R + size_v * R :].reshape(size_w, R)
        
        T = tensor_contraction(U, V, W)
        return np.sum((T - TARGET_TENSOR)**2)

    # Schoolbook solution as fallback
    best_rank = n * m * p # 48
    U_school = np.zeros((size_u, best_rank))
    V_school = np.zeros((size_v, best_rank))
    W_school = np.zeros((size_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_solution = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # Try to find better ranks
    # Start from 47 down to 40
    for R in range(47, 39, -1):
        if time.time() > deadline - 5:
            break
            
        # Initialize parameters
        # Use a few different random seeds
        for seed in range(3):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + R + seed)
            params = np.random.randn(size_u * R + size_v * R + size_w * R) * 0.5
            
            try:
                res = minimize(objective, params, method='L-BFGS-B', 
                               options={'maxiter': 20, 'maxfun': 200})
                
                if res.fun < 1e-6:
                    R_opt = R
                    U_opt = res.x[:size_u * R_opt].reshape(size_u, R_opt)
                    V_opt = res.x[size_u * R_opt : size_u * R_opt + size_v * R_opt].reshape(size_v, R_opt)
                    W_opt = res.x[size_u * R_opt + size_v * R_opt :].reshape(size_w, R_opt)
                    
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    T_snap = tensor_contraction(U_snap, V_snap, W_snap)
                    if np.allclose(T_snap, TARGET_TENSOR, atol=1e-9):
                        best_rank = R
                        best_solution = {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        break # Found a solution for this R, try lower R? 
                             # No, we want MINIMAL rank. We are iterating down. 
                             # If we find R=47, it is better than 48. 
                             # Should we break the outer loop? 
                             # Yes, because we found a valid solution with rank R. 
                             # If we continue, we might find R=46, which is even better. 
                             # But finding R=46 is harder. 
                             # Let's break the inner loop and continue to lower R.
                # If we found a solution, we should try to find an even lower rank.
                # So we don't break the outer loop immediately, but we update best_solution.
                # However, if we find R=47, we should still try R=46.
                
            except Exception:
                pass
        
        # If we found a solution for this R, we can continue to lower R.
        # If we didn't, we continue to lower R.
        # So no break in outer loop unless time is up.

    return best_solution

# EVOLVE-BLOCK-END
