import scipy
import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    n, m, p = 3, 3, 6
    
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Construct target tensor T_target
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def snap_and_verify(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        R = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        if np.allclose(R, T_target, atol=1e-9):
            return True, U_s.tolist(), V_s.tolist(), W_s.tolist()
        return False, None, None, None

    def get_schoolbook(rank):
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Strategy: Try to reduce rank by finding linear dependencies
    # Start from schoolbook and try to eliminate one term at a time
    
    # First, let's try a direct approach: for each candidate rank r < 54,
    # try to find a decomposition using structured random search with better initialization
    
    # Better initialization: use the schoolbook structure as a starting point
    # and apply small perturbations, then optimize
    
    def compute_residual(U, V, W, rank):
        R = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((R - T_target)**2)

    # Try ranks from 53 down to 50 (more realistic range)
    for target_rank in range(53, 49, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple structured initializations
        num_attempts = 5
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize close to schoolbook with some columns merged
            U = np.zeros((dim_U, target_rank))
            V = np.zeros((dim_V, target_rank))
            W = np.zeros((dim_W, target_rank))
            
            # Use first target_rank schoolbook terms
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < target_rank:
                            U[i*m+j, col] = 1
                            V[j*p+k, col] = 1
                            W[k*n+i, col] = 1
                            col += 1
                        else:
                            break
                    if col >= target_rank:
                        break
                if col >= target_rank:
                    break
            
            # Add small random perturbations to all columns
            np.random.seed(100 + target_rank * 10 + attempt)
            U = U + np.random.randn(dim_U, target_rank) * 0.3
            V = V + np.random.randn(dim_V, target_rank) * 0.3
            W = W + np.random.randn(dim_W, target_rank) * 0.3
            
            # Optimize
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def obj(params, target_rank=target_rank):
                U_p = params[:dim_U*target_rank].reshape(dim_U, target_rank)
                V_p = params[dim_U*target_rank : dim_U*target_rank + dim_V*target_rank].reshape(dim_V, target_rank)
                W_p = params[dim_U*target_rank + dim_V*target_rank :].reshape(dim_W, target_rank)
                return compute_residual(U_p, V_p, W_p, target_rank)
            
            try:
                res = scipy.optimize.minimize(
                    obj, params, method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                U_opt = res.x[:dim_U*target_rank].reshape(dim_U, target_rank)
                V_opt = res.x[dim_U*target_rank : dim_U*target_rank + dim_V*target_rank].reshape(dim_V, target_rank)
                W_opt = res.x[dim_U*target_rank + dim_V*target_rank :].reshape(dim_W, target_rank)
                
                success, U_s, V_s, W_s = snap_and_verify(U_opt, V_opt, W_opt)
                if success:
                    return {"U": U_s, "V": V_s, "W": W_s}
                    
            except Exception:
                pass

    # Fallback to schoolbook
    U, V, W = get_schoolbook(54)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
