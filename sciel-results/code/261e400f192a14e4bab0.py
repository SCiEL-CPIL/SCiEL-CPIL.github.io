import numpy as np
import time

def construct():
    deadline = time.time() + 140
    n, m, p = 3, 4, 7
    
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    def verify(U, V, W, tol=1e-6):
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for j in range(m):
                    u_idx = i * m + j
                    for j_prime in range(m):
                        v_idx = j_prime * p + k
                        val = np.sum(W[w_idx, :] * U[u_idx, :] * V[v_idx, :])
                        target = 1.0 if j == j_prime else 0.0
                        err = abs(val - target)
                        if err > max_err:
                            max_err = err
        return max_err < tol
    
    def residual(params, r):
        U = params[0:r*dim_U].reshape(r, dim_U).T
        V = params[r*dim_U:r*(dim_U+dim_V)].reshape(r, dim_V).T
        W = params[r*(dim_U+dim_V):].reshape(r, dim_W).T
        
        res = np.zeros(n * p * m * m)
        idx = 0
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                for j in range(m):
                    u_idx = i * m + j
                    for j_prime in range(m):
                        v_idx = j_prime * p + k
                        val = np.sum(W[w_idx, :] * U[u_idx, :] * V[v_idx, :])
                        target = 1.0 if j == j_prime else 0.0
                        res[idx] = val - target
                        idx += 1
        return res
    
    def try_rank(r, n_restarts=3, max_iter=500):
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                return None
            np.random.seed(12345 + restart * 100 + r)
            scale = 0.5
            U_init = np.random.randn(dim_U, r) * scale
            V_init = np.random.randn(dim_V, r) * scale
            W_init = np.random.randn(dim_W, r) * scale
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            from scipy.optimize import least_squares
            try:
                sol = least_squares(residual, params_init, args=(r,), method='lm', max_nfev=max_iter)
                if sol.success and np.max(np.abs(sol.fun)) < 1e-5:
                    U_sol = sol.x[0:r*dim_U].reshape(r, dim_U).T
                    V_sol = sol.x[r*dim_U:r*(dim_U+dim_V)].reshape(r, dim_V).T
                    W_sol = sol.x[r*(dim_U+dim_V):].reshape(r, dim_W).T
                    
                    if verify(U_sol, V_sol, W_sol, tol=1e-4):
                        return (U_sol, V_sol, W_sol)
            except Exception:
                continue
        return None
    
    # Try ranks from 19 upward
    for r in range(19, 35):
        if time.time() > deadline - 15:
            break
        result = try_rank(r, n_restarts=3, max_iter=400)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback: Schoolbook decomposition (Rank 84)
    rank_school = n * m * p
    U = np.zeros((dim_U, rank_school))
    V = np.zeros((dim_V, rank_school))
    W = np.zeros((dim_W, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
