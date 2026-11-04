import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    deadline = time.time() + 110
    n, m, p = 3, 4, 7
    
    # Dimensions of U, V, W
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    # Precompute index arrays for vectorized residual
    # For each (i, k, j, j_prime) tuple, we need indices into U, V, W
    i_vals = []
    k_vals = []
    j_vals = []
    jp_vals = []
    targets = []
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    i_vals.append(i)
                    k_vals.append(k)
                    j_vals.append(j)
                    jp_vals.append(jp)
                    targets.append(1.0 if j == jp else 0.0)
    
    i_arr = np.array(i_vals)
    k_arr = np.array(k_vals)
    j_arr = np.array(j_vals)
    jp_arr = np.array(jp_vals)
    target_arr = np.array(targets)
    
    # Precompute linear indices
    w_idx_arr = k_arr * n + i_arr  # index into W
    u_idx_arr = i_arr * m + j_arr  # index into U
    v_idx_arr = jp_arr * p + k_arr  # index into V
    
    def verify(U, V, W, tol=1e-5):
        r = U.shape[1]
        # Vectorized check
        # W[w_idx, r] * U[u_idx, r] * V[v_idx, r] summed over r
        vals = np.einsum('ir,jr,kr->ijk', 
                         W[w_idx_arr, :], 
                         U[u_idx_arr, :], 
                         V[v_idx_arr, :])
        max_err = np.max(np.abs(vals - target_arr))
        return max_err < tol

    def residual(params, r):
        # params is a flat array of length r * (dim_U + dim_V + dim_W)
        U = params[0:r*dim_U].reshape(dim_U, r)
        V = params[r*dim_U:r*(dim_U+dim_V)].reshape(dim_V, r)
        W = params[r*(dim_U+dim_V):].reshape(dim_W, r)
        
        # Vectorized residual
        vals = np.einsum('ir,jr,kr->ijk', 
                         W[w_idx_arr, :], 
                         U[u_idx_arr, :], 
                         V[v_idx_arr, :])
        return vals - target_arr

    # Try ranks: known lower bounds suggest ~49-56 for 3x4x7
    # Start from lower ranks to find minimal solution faster
    for r in range(50, 45, -1):
        if time.time() > deadline - 15:
            break
            
        # Multiple random restarts with different scales
        for restart in range(10):
            if time.time() > deadline - 5:
                break
                
            seed = 42 + restart * 1000
            np.random.seed(seed)
            
            # Use moderate scale initialization
            scale = 0.5 + 0.1 * restart
            U_init = np.random.randn(dim_U, r) * scale
            V_init = np.random.randn(dim_V, r) * scale
            W_init = np.random.randn(dim_W, r) * scale
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                sol = least_squares(residual, params_init, args=(r,), method='lm', max_nfev=2000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                max_res = np.max(np.abs(sol.fun))
                if max_res < 1e-6:
                    U_sol = sol.x[0:r*dim_U].reshape(dim_U, r)
                    V_sol = sol.x[r*dim_U:r*(dim_U+dim_V)].reshape(dim_V, r)
                    W_sol = sol.x[r*(dim_U+dim_V):].reshape(dim_W, r)
                    
                    # Final verification with tight tolerance
                    if verify(U_sol, V_sol, W_sol, tol=1e-5):
                        return {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
            except Exception:
                continue

    # If we haven't found a solution in 50-46, try a few more ranks
    for r in range(45, 40, -1):
        if time.time() > deadline - 10:
            break
            
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            seed = 10000 + restart * 7777
            np.random.seed(seed)
            
            scale = 0.3 + 0.05 * restart
            U_init = np.random.randn(dim_U, r) * scale
            V_init = np.random.randn(dim_V, r) * scale
            W_init = np.random.randn(dim_W, r) * scale
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                sol = least_squares(residual, params_init, args=(r,), method='lm', max_nfev=1500, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                max_res = np.max(np.abs(sol.fun))
                if max_res < 1e-6:
                    U_sol = sol.x[0:r*dim_U].reshape(dim_U, r)
                    V_sol = sol.x[r*dim_U:r*(dim_U+dim_V)].reshape(dim_V, r)
                    W_sol = sol.x[r*(dim_U+dim_V):].reshape(dim_W, r)
                    
                    if verify(U_sol, V_sol, W_sol, tol=1e-5):
                        return {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
            except Exception:
                continue

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
