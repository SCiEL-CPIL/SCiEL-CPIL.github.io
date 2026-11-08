import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Fallback: Schoolbook decomposition (Rank 24)
    def get_schoolbook():
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return R, U, V, W

    R_school, U_school, V_school, W_school = get_schoolbook()
    
    # Try to find a better rank
    best_result = (R_school, U_school, V_school, W_school)
    
    # Try ranks from 23 down to 14
    for R in range(23, 13, -1):
        if time.time() > deadline - 10: # Leave 10s for verification/fallback
            break
            
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 100 + R)
            # Initialize
            U_flat = np.random.uniform(-1, 1, (n * m * R))
            V_flat = np.random.uniform(-1, 1, (m * p * R))
            W_flat = np.random.uniform(-1, 1, (p * n * R))
            
            U = U_flat.reshape(n, m, R)
            V = V_flat.reshape(m, p, R)
            W = W_flat.reshape(p, n, R)
            
            def residuals(params):
                U_p = params[0:n*m*R].reshape(n, m, R)
                V_p = params[n*m*R:n*m*R + m*p*R].reshape(m, p, R)
                W_p = params[n*m*R + m*p*R:].reshape(p, n, R)
                
                res = []
                for i in range(n):
                    for k in range(p):
                        M = np.zeros((m, m))
                        for r in range(R):
                            M += W_p[k, i, r] * np.outer(U_p[i, :, r], V_p[:, k, r])
                        res.extend((M - np.eye(m)).flatten())
                return np.array(res)
            
            x0 = np.concatenate([U_flat, V_flat, W_flat])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=200)
                if np.linalg.norm(result.fun) < 1e-5:
                    U_sol = result.x[0:n*m*R].reshape(n, m, R)
                    V_sol = result.x[n*m*R:n*m*R + m*p*R].reshape(m, p, R)
                    W_sol = result.x[n*m*R + m*p*R:].reshape(p, n, R)
                    
                    # Try rounding to half-integers
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    if _check_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                        best_result = (R, U_int.tolist(), V_int.tolist(), W_int.tolist())
                        break
            except Exception:
                pass
        if best_result[0] < R_school:
            break
            
    R, U, V, W = best_result
    return {"U": U, "V": V, "W": W}

def _check_decomposition(U, V, W, n, m, p):
    R = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i * m + j][r] * V[jp * p + k][r] * W[k * n + i][r]
                    target = 1 if j == jp else 0
                    if not np.isclose(s, target, atol=1e-9):
                        return False
    return True
