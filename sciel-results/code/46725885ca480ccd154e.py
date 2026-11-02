"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_tensors = n * m * m * p
    T_target = np.zeros(num_tensors)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    T_target[idx] = 1.0 if j == k else 0.0
                    idx += 1

    def compute_residuals(x, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U + size_V]
        W_flat = x[size_U + size_V:]
        
        U = U_flat.reshape(n * m, R)
        V = V_flat.reshape(m * p, R)
        W = W_flat.reshape(p * n, R)
        
        residuals = np.zeros(num_tensors)
        idx = 0
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(m):
                    v_row = V[j * p + k]
                    for l in range(p):
                        w_row = W[k * n + l]
                        val = np.sum(u_row * v_row * w_row)
                        residuals[idx] = val - T_target[idx]
                        idx += 1
        return residuals

    def try_find_decomposition(R, num_trials=50):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        for trial in range(num_trials):
            np.random.seed(42 + trial)
            x0 = np.random.randn(total_size) * 0.5
            
            try:
                res = least_squares(compute_residuals, x0, args=(R,), method='trf', max_nfev=20000)
            except Exception:
                continue
            
            if res.cost < 1e-8:
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U = U_flat.reshape(n * m, R)
                V = V_flat.reshape(m * p, R)
                W = W_flat.reshape(p * n, R)
                
                U_2 = 2.0 * U
                V_2 = 2.0 * V
                W_2 = 2.0 * W
                
                U_int = np.round(U_2)
                V_int = np.round(V_2)
                W_int = np.round(W_2)
                
                if np.max(np.abs(U_int - U_2)) < 1e-6 and \
                   np.max(np.abs(V_int - V_2)) < 1e-6 and \
                   np.max(np.abs(W_int - W_2)) < 1e-6:
                    
                    U_final = U_int / 2.0
                    V_final = V_int / 2.0
                    W_final = W_int / 2.0
                    
                    x_final = np.concatenate([U_final.flatten(), V_final.flatten(), W_final.flatten()])
                    res_final = compute_residuals(x_final, R)
                    if np.max(np.abs(res_final)) < 1e-9:
                        return {
                            "U": U_final.tolist(),
                            "V": V_final.tolist(),
                            "W": W_final.tolist()
                        }
        return None

    for R in range(8, 13):
        result = try_find_decomposition(R)
        if result is not None:
            return result

    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + j][column] = 1.0
                W[j * n + l][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
