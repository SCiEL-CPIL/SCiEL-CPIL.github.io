import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 6]

def construct():
    deadline = time.time() + 140.0
    
    n, m, p = DIMENSIONS
    
    def check_decomp(U, V, W):
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

    def residuals_333(params, r):
        total_size = (9 + 9 + 9) * r
        if len(params) != total_size:
            return None
        
        U = params[0:9*r].reshape(9, r)
        V = params[9*r:18*r].reshape(9, r)
        W = params[18*r:27*r].reshape(9, r)
        
        res = np.zeros(27)
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u_idx = i * 3 + j
                    v_idx = j * 3 + k
                    w_idx = k * 3 + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res

    def verify_333(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u_idx = i * 3 + j
                    v_idx = j * 3 + k
                    w_idx = k * 3 + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if not np.isclose(val, 1.0, atol=1e-8):
                        return False
        return True

    def find_333_decomp(rank, max_iter=30, timeout=60):
        start_time = time.time()
        for attempt in range(max_iter):
            if time.time() > start_time + timeout or time.time() > deadline:
                return None
            
            r = rank
            size_U = 9 * r
            size_V = 9 * r
            size_W = 9 * r
            
            params = np.random.randn(size_U + size_V + size_W) * 0.5
            
            try:
                result = least_squares(residuals_333, params, args=(rank,), method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12)
            except Exception:
                continue
                
            if result.success and result.cost < 1e-10:
                U = result.x[0:size_U].reshape(9, rank)
                V = result.x[size_U:size_U+size_V].reshape(9, rank)
                W = result.x[size_U+size_V:size_U+size_V+size_W].reshape(9, rank)
                
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify_333(U_r, V_r, W_r):
                    return (U_r, V_r, W_r)
                
                params2 = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
                try:
                    result2 = least_squares(residuals_333, params2, args=(rank,), method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12)
                    if result2.success and result2.cost < 1e-10:
                        U2 = result2.x[0:size_U].reshape(9, rank)
                        V2 = result2.x[size_U:size_U+size_V].reshape(9, rank)
                        W2 = result2.x[size_U+size_V:size_U+size_V+size_W].reshape(9, rank)
                        
                        U_r2 = np.round(U2 * 2) / 2
                        V_r2 = np.round(V2 * 2) / 2
                        W_r2 = np.round(W2 * 2) / 2
                        
                        if verify_333(U_r2, V_r2, W_r2):
                            return (U_r2, V_r2, W_r2)
                except Exception:
                    pass
        
        return None

    def build_336_from_333(U3, V3, W3):
        r3 = U3.shape[1]
        r6 = 2 * r3
        
        U6 = np.zeros((9, r6))
        V6 = np.zeros((18, r6))
        W6 = np.zeros((18, r6))
        
        U6[:, :r3] = U3
        U6[:, r3:] = U3
        
        for j in range(3):
            for k in range(3):
                V6[j*6+k, :r3] = V3[j*3+k, :]
                V6[j*6+k+3, r3:] = V3[j*3+k, :]
        
        for k in range(3):
            for i in range(3):
                W6[k*3+i, :r3] = W3[k*3+i, :]
                W6[(k+3)*3+i, r3:] = W3[k*3+i, :]
        
        return U6, V6, W6

    for target_rank in [22, 23, 24]:
        if time.time() > deadline - 20:
            break
        decomp333 = find_333_decomp(target_rank, max_iter=15, timeout=50)
        if decomp333 is not None:
            U3, V3, W3 = decomp333
            U6, V6, W6 = build_336_from_333(U3, V3, W3)
            
            if check_decomp(U6, V6, W6):
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    
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
