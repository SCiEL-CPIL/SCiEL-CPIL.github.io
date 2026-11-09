import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 9
    dim_v = m * p  # 15
    dim_w = p * n  # 15
    
    # Build target tensor
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    # Schoolbook solution as fallback
    best_rank = n * m * p  # 45
    U_sb = np.zeros((dim_u, best_rank))
    V_sb = np.zeros((dim_v, best_rank))
    W_sb = np.zeros((dim_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
    
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    start_time = time.time()
    time_limit = 140  # Safety margin before 150s kill
    
    # Try to find lower rank
    for R in range(best_rank - 1, 29, -1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        # Try multiple random seeds
        for seed in range(30):
            if time.time() - start_time > time_limit:
                break
            
            np.random.seed(seed * 1000 + R)
            U_init = np.random.randn(dim_u, R) * 0.5
            V_init = np.random.randn(dim_v, R) * 0.5
            W_init = np.random.randn(dim_w, R) * 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def res_func(params, R=R):
                U = params[:dim_u*R].reshape(dim_u, R)
                V = params[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
                W = params[dim_u*R + dim_v*R:].reshape(dim_w, R)
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                return (T_recon - target).flatten()
            
            try:
                result = least_squares(
                    res_func, 
                    params_init, 
                    method='lm', 
                    max_nfev=5000,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12
                )
                
                if result.cost < 1e-8:
                    params = result.x
                    U = params[:dim_u*R].reshape(dim_u, R)
                    V = params[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
                    W = params[dim_u*R + dim_v*R:].reshape(dim_w, R)
                    
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify all entries
                    T_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                    if np.allclose(T_snap, target, atol=1e-9):
                        best_rank = R
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
