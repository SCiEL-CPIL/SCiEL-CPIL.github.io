import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Precompute target tensor flat
    T_target = np.zeros(dim_U * dim_V * dim_W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # Index in flat array: (u_idx * dim_V + v_idx) * dim_W + w_idx
                idx = (u_idx * dim_V + v_idx) * dim_W + w_idx
                T_target[idx] = 1.0
                
    def residual(params, r):
        # params length r * (dim_U + dim_V + dim_W)
        U_flat = params[:r*dim_U]
        V_flat = params[r*dim_U:r*(dim_U+dim_V)]
        W_flat = params[r*(dim_U+dim_V):]
        
        # Reshape to (dim, r)
        U = U_flat.reshape(dim_U, r)
        V = V_flat.reshape(dim_V, r)
        W = W_flat.reshape(dim_W, r)
        
        # Compute T_recon
        # T_recon[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
        # This is a tensor product sum.
        # We can do:
        # T_recon = np.einsum('uc,vc,wc->uvw', U, V, W)
        # But einsum might be slow for small arrays?
        # Let's try loop for clarity and speed if r is small
        T_recon = np.zeros((dim_U, dim_V, dim_W))
        for c in range(r):
            T_recon += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
            
        return T_recon.flatten() - T_target

    start_time = time.time()
    time_limit = 140.0
    
    # Try ranks from 23 down to 15
    # If we find a valid one, return it
    for r in range(23, 14, -1):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Number of restarts based on time remaining
        n_restarts = 3
        
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Random init
            np.random.seed(123 + r + restart)
            params = np.random.randn(r * (dim_U + dim_V + dim_W))
            
            from scipy.optimize import least_squares
            
            def fun(x):
                return residual(x, r)
                
            try:
                res = least_squares(fun, params, method='trf', max_nfev=200)
            except Exception as e:
                continue
                
            if res.success or res.cost < 1e-8:
                # Check rounding
                U_flat = res.x[:r*dim_U]
                V_flat = res.x[r*dim_U:r*(dim_U+dim_V)]
                W_flat = res.x[r*(dim_U+dim_V):]
                
                U_rounded = np.round(U_flat * 2) / 2
                V_rounded = np.round(V_flat * 2) / 2
                W_rounded = np.round(W_flat * 2) / 2
                
                # Verify
                check_params = np.concatenate([U_rounded, V_rounded, W_rounded])
                res_check = residual(check_params, r)
                
                if np.max(np.abs(res_check)) < 1e-9:
                    U_mat = U_rounded.reshape(dim_U, r)
                    V_mat = V_rounded.reshape(dim_V, r)
                    W_mat = W_rounded.reshape(dim_W, r)
                    
                    return {
                        "U": U_mat.tolist(),
                        "V": V_mat.tolist(),
                        "W": W_mat.tolist()
                    }

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
