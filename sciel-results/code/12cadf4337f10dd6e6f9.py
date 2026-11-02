import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 6]

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    def check_decomp(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if not np.isclose(val, 1.0, atol=1e-8):
                        return False
        return True

    def residuals(params, r):
        total_size = (9 + 18 + 18) * r
        if len(params) != total_size:
            return None
        
        U = params[0:9*r].reshape(9, r)
        V = params[9*r:9*r+18*r].reshape(18, r)
        W = params[9*r+18*r:9*r+18*r+18*r].reshape(18, r)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    def find_decomp(rank, max_iter=10, timeout=100):
        start_time = time.time()
        for attempt in range(max_iter):
            if time.time() > start_time + timeout:
                return None
            
            r = rank
            size_U = 9 * r
            size_V = 18 * r
            size_W = 18 * r
            
            # Random init
            params = np.random.randn(size_U + size_V + size_W) * 0.5
            
            try:
                result = least_squares(residuals, params, args=(rank,), method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12)
            except Exception:
                continue
                
            if result.success and result.cost < 1e-10:
                U = result.x[0:size_U].reshape(9, rank)
                V = result.x[size_U:size_U+size_V].reshape(18, rank)
                W = result.x[size_U+size_V:size_U+size_V+size_W].reshape(18, rank)
                
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if check_decomp(U_r, V_r, W_r):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
                # Polish: use rounded values as new initial guess
                params2 = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
                try:
                    result2 = least_squares(residuals, params2, args=(rank,), method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12)
                    if result2.success and result2.cost < 1e-10:
                        U2 = result2.x[0:size_U].reshape(9, rank)
                        V2 = result2.x[size_U:size_U+size_V].reshape(18, rank)
                        W2 = result2.x[size_U+size_V:size_U+size_V+size_W].reshape(18, rank)
                        
                        U_r2 = np.round(U2 * 2) / 2
                        V_r2 = np.round(V2 * 2) / 2
                        W_r2 = np.round(W2 * 2) / 2
                        
                        if check_decomp(U_r2, V_r2, W_r2):
                            return {"U": U_r2.tolist(), "V": V_r2.tolist(), "W": W_r2.tolist()}
                except Exception:
                    pass
        
        return None

    # Try rank 44
    decomp = find_decomp(44, max_iter=10, timeout=100)
    if decomp:
        return decomp

    # Fallback to schoolbook 54
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
