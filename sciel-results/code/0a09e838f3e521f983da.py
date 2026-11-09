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
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) else 0
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
                
    # Schoolbook solution as fallback (rank = n*m*p = 45)
    best_rank = n * m * p
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
    
    best_U = None
    best_V = None
    best_W = None
    best_rank_found = None
    
    # Time management
    start_time = time.time()
    time_limit = 110  # Safety margin before 120s limit
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the target tensor."""
        try:
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            return np.allclose(T_recon, target, atol=1e-9)
        except Exception:
            return False
    
    def try_rank(R, U_init, V_init, W_init, max_nfev=3000):
        """Try to find a valid decomposition at rank R."""
        def res_func(params, R=R):
            U = params[:dim_u*R].reshape(dim_u, R)
            V = params[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
            W = params[dim_u*R + dim_v*R:].reshape(dim_w, R)
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            return (T_recon - target).flatten()
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = least_squares(
                res_func, 
                x0, 
                method='lm', 
                max_nfev=max_nfev
            )
            
            if result.success or result.cost < 1e-6:
                params = result.x
                U = params[:dim_u*R].reshape(dim_u, R)
                V = params[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
                W = params[dim_u*R + dim_v*R:].reshape(dim_w, R)
                
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if verify_decomposition(U_snap, V_snap, W_snap):
                    return U_snap, V_snap, W_snap
                
                # If snap fails, check if original is valid
                if verify_decomposition(U, V, W):
                    # Round to nearest integer or half-integer
                    U_out = np.round(U * 2) / 2
                    V_out = np.round(V * 2) / 2
                    W_out = np.round(W * 2) / 2
                    if verify_decomposition(U_out, V_out, W_out):
                        return U_out, V_out, W_out
        except Exception:
            pass
        
        return None
    
    # Strategy: Try ranks from 44 down to 35
    # For each rank, try multiple random initializations
    for R in range(best_rank - 1, 34, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random initializations for this rank
        found = False
        num_attempts = min(5, max(1, int((time_limit - (time.time() - start_time)) / 10)))
        
        for attempt in range(num_attempts):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(42 + R * 100 + attempt)
            U_init = np.random.randn(dim_u, R) * 0.5
            V_init = np.random.randn(dim_v, R) * 0.5
            W_init = np.random.randn(dim_w, R) * 0.5
            
            result = try_rank(R, U_init, V_init, W_init, max_nfev=2000)
            if result is not None:
                U_snap, V_snap, W_snap = result
                best_rank_found = R
                best_U = U_snap
                best_V = V_snap
                best_W = W_snap
                found = True
                break
        
        if found:
            # Found a valid decomposition at this rank, try lower ranks
            continue
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
