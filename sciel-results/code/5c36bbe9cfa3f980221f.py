import numpy as np
import time
import sys
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 3, 4
    dim_u = n * m  # 6
    dim_v = m * p  # 12
    dim_w = p * n  # 8
    
    # Create target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
    
    T_flat = T.flatten()
    
    lower_bound = max(dim_u, dim_v, dim_w)
    
    def objective(x, R):
        U = x[:dim_u * R].reshape(dim_u, R)
        V = x[dim_u * R : dim_u * R + dim_v * R].reshape(dim_v, R)
        W = x[dim_u * R + dim_v * R :].reshape(dim_w, R)
        X = np.einsum('ir,jr,kr->ijk', U, V, W)
        return (X.flatten() - T_flat)
    
    def try_rank(R, n_restarts=20):
        for restart in range(n_restarts):
            # Use different seeds
            np.random.seed(restart * 7919 + R * 31)
            U0 = np.random.randn(dim_u, R) * 0.5
            V0 = np.random.randn(dim_v, R) * 0.5
            W0 = np.random.randn(dim_w, R) * 0.5
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            try:
                # Use 'lm' for least squares
                res = least_squares(lambda x: objective(x, R), x0, method='lm', max_nfev=5000)
                if res.success and res.cost < 1e-6:
                    x_opt = res.x
                    U_opt = x_opt[:dim_u * R].reshape(dim_u, R)
                    V_opt = x_opt[dim_u * R : dim_u * R + dim_v * R].reshape(dim_v, R)
                    W_opt = x_opt[dim_u * R + dim_v * R :].reshape(dim_w, R)
                    
                    # Snap to 0.5 grid
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    # Verify exactness
                    X_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                    
                    if np.allclose(X_snap, T, atol=1e-9):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
            except Exception:
                pass
        return None

    start_time = time.time()
    deadline = start_time + 140
    
    # Try ranks from lower bound to 23
    for R in range(lower_bound, 24):
        if time.time() > deadline:
            break
            
        # Reduce restarts if time is running out
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        result = try_rank(R, n_restarts=20)
        if result is not None:
            return result
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][column] = 1.0
                V[v_idx][column] = 1.0
                W[w_idx][column] = 1.0
                column += 1
                
    return {"U": U, "V": V, "W": W}
