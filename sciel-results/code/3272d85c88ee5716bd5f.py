import numpy as np
import time
from scipy.optimize import least_squares, minimize

n, m, p = 2, 5, 6

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def check_decomposition(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_optimize_rank(R):
        """
        Try to find a decomposition of rank R using continuous optimization.
        """
        if time.time() > deadline - 20:
            return None
            
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        def residuals(x):
            U = x[:size_U].reshape((n*m, R))
            V = x[size_U:size_U+size_V].reshape((m*p, R))
            W = x[size_U+size_V:].reshape((p*n, R))
            
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                        res[idx] = val - 1.0
                        idx += 1
            return res

        # Try multiple initialization strategies
        init_strategies = []
        
        # Strategy 1: Random
        for _ in range(5):
            init_strategies.append(np.random.randn(total_size) * 0.5)
        
        # Strategy 2: Near schoolbook (if R is close to 60)
        if R >= 55:
            U_s, V_s, W_s = get_schoolbook()
            # Take first R columns
            U_init = U_s[:, :R].flatten()
            V_init = V_s[:, :R].flatten()
            W_init = W_s[:, :R].flatten()
            init_strategies.append(np.concatenate([U_init, V_init, W_init]))
        
        for x0 in init_strategies:
            if time.time() > deadline - 15:
                return None
                
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            except Exception:
                continue
                
            if result.cost < 1e-10:
                U = result.x[:size_U].reshape((n*m, R))
                V = result.x[size_U:size_U+size_V].reshape((m*p, R))
                W = result.x[size_U+size_V:].reshape((p*n, R))
                
                # Try to snap to integers/half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                if check_decomposition(U_int, V_int, W_int):
                    return U_int, V_int, W_int
                
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                if check_decomposition(U_int, V_int, W_int):
                    return U_int, V_int, W_int
        return None

    # Try ranks from 59 down to 50
    for R in range(59, 49, -1):
        if time.time() > deadline - 30:
            break
        result = try_optimize_rank(R)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
