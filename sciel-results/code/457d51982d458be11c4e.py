import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    target_rank = 14
    
    # Dimensions of U, V, W
    u_rows = n * m
    v_rows = m * p
    w_rows = p * n
    
    # Target Tensor T
    # T[u, v, w] = 1 if indices match, else 0
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    T = np.zeros((u_rows, v_rows, w_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def residual_func(params):
        # params: flattened U, V, W
        # U: u_rows x rank
        # V: v_rows x rank
        # W: w_rows x rank
        u_flat, v_flat, w_flat = params[:u_rows*target_rank], params[u_rows*target_rank:u_rows*target_rank + v_rows*target_rank], params[u_rows*target_rank + v_rows*target_rank:]
        
        U = u_flat.reshape((u_rows, target_rank))
        V = v_flat.reshape((v_rows, target_rank))
        W = w_flat.reshape((w_rows, target_rank))
        
        # Compute sum_c U[u,c] V[v,c] W[w,c] for all u,v,w
        # This is a 3-way tensor contraction
        # Residual[u,v,w] = sum_c U[u,c] V[v,c] W[w,c] - T[u,v,w]
        
        # Efficient computation:
        # Let M = U @ V.T (u_rows x v_rows) ? No, that sums over c.
        # We need to keep c dimension? No, we sum over c.
        # sum_c U[u,c] V[v,c] W[w,c]
        # This is not a simple matrix product.
        # It is sum_c (U[:,c] outer V[:,c]) * W[:,c] ? No.
        # It is sum_c U[u,c] * V[v,c] * W[w,c].
        
        # We can compute this by iterating over c or using einsum.
        # np.einsum('uc,vc,wc->uvw', U, V, W)
        
        try:
            calc = np.einsum('uc,vc,wc->uvw', U, V, W)
        except Exception:
            # Fallback if einsum fails or is too slow?
            calc = np.zeros_like(T)
            for c in range(target_rank):
                calc += np.einsum('u,v,w->uvw', U[:,c], V[:,c], W[:,c])
                
        return (calc - T).flatten()

    start_time = time.time()
    deadline = start_time + 100  # Leave 20s for cleanup
    
    best_params = None
    best_res = np.inf
    
    # Random restarts
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scale matters. Let's try small random values.
        np.random.seed(i)
        U_init = np.random.randn(u_rows, target_rank) * 0.1
        V_init = np.random.randn(v_rows, target_rank) * 0.1
        W_init = np.random.randn(w_rows, target_rank) * 0.1
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = least_squares(residual_func, params_init, method='lm', max_nfev=1000)
            if res.cost < best_res:
                best_res = res.cost
                best_params = res.x
                if best_res < 1e-10:
                    break
        except Exception:
            pass
            
    if best_params is not None and best_res < 1e-10:
        U_flat, V_flat, W_flat = best_params[:u_rows*target_rank], best_params[u_rows*target_rank:u_rows*target_rank + v_rows*target_rank], best_params[u_rows*target_rank + v_rows*target_rank:]
        U = U_flat.reshape((u_rows, target_rank))
        V = V_flat.reshape((v_rows, target_rank))
        W = W_flat.reshape((w_rows, target_rank))
        
        # Convert to list of lists
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
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
