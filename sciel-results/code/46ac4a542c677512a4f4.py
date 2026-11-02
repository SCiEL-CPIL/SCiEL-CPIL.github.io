import numpy as np
import time
import sys
from scipy.optimize import minimize, least_squares

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
    
    def check_and_snap(U, V, W):
        # Snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        T_snap = tensor_contraction(U_snap, V_snap, W_snap)
        if np.allclose(T_snap, TARGET_TENSOR, atol=1e-9):
            return True, U_snap, V_snap, W_snap
        return False, None, None, None

    # Try to find better ranks
    # Expand search range to 32
    for R in range(47, 31, -1):
        if time.time() > deadline - 5:
            break
            
        num_restarts = 5
        for seed in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + R * 100 + seed)
            # Scale initialization to 1/sqrt(R) to keep contraction terms O(1)
            scale = 1.0 / np.sqrt(R)
            params = np.random.randn(size_u * R + size_v * R + size_w * R) * scale
            
            def objective(params):
                if time.time() > deadline:
                    return 1e10
                
                U = params[:size_u * R].reshape(size_u, R)
                V = params[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
                W = params[size_u * R + size_v * R :].reshape(size_w, R)
                
                T = tensor_contraction(U, V, W)
                return np.sum((T - TARGET_TENSOR)**2)

            def residuals(params):
                if time.time() > deadline:
                    return np.ones(size_u * size_v * size_w)
                
                U = params[:size_u * R].reshape(size_u, R)
                V = params[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
                W = params[size_u * R + size_v * R :].reshape(size_w, R)
                
                T = tensor_contraction(U, V, W)
                return (T - TARGET_TENSOR).flatten()

            try:
                # Try least_squares first as it's designed for this
                res = least_squares(residuals, params, method='trf', 
                                    options={'max_nfev': 200})
                
                if res.cost < 1e-6:
                    U_opt = res.x[:size_u * R].reshape(size_u, R)
                    V_opt = res.x[size_u * R : size_u * R + size_v * R].reshape(size_v, R)
                    W_opt = res.x[size_u * R + size_v * R :].reshape(size_w, R)
                    
                    valid, U_snap, V_snap, W_snap = check_and_snap(U_opt, V_opt, W_opt)
                    if valid:
                        best_rank = R
                        best_solution = {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        break # Move to lower R
                        
            except Exception:
                pass
        
    return best_solution

# EVOLVE-BLOCK-END
