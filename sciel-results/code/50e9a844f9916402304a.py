import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    num_U = n * m
    num_V = m * p
    num_W = p * n
    
    # Pre-compute indices for the tensor equations
    # Equation for (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    i_vals, j_vals, k_vals = np.meshgrid(range(n), range(m), range(p), indexing='ij')
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    u_idx = i_flat * m + j_flat
    v_idx = j_flat * p + k_flat
    w_idx = k_flat * n + i_flat
    
    def residual(params, r):
        size_U = num_U * r
        size_V = num_V * r
        size_W = num_W * r
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(num_U, r)
        V = V_flat.reshape(num_V, r)
        W = W_flat.reshape(num_W, r)
        
        U_sel = U[u_idx, :]
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        
        products = U_sel * V_sel * W_sel
        sums = np.sum(products, axis=1)
        
        return sums - 1

    def objective(params, r):
        res = residual(params, r)
        return np.sum(res * res)
        
    def verify_and_round(U, V, W, r):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        U_sel = U_r[u_idx, :]
        V_sel = V_r[v_idx, :]
        W_sel = W_r[w_idx, :]
        sums = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        if np.allclose(sums, 1.0, atol=1e-9):
            return U_r.tolist(), V_r.tolist(), W_r.tolist()
        return None

    def try_rank(r, time_limit=15.0):
        start_time = time.time()
        
        for attempt in range(10):
            if time.time() - start_time > time_limit:
                break
                
            scale = 1.0 / (r ** 0.5)
            
            U_init = np.random.randn(num_U, r) * scale
            V_init = np.random.randn(num_V, r) * scale
            W_init = np.random.randn(num_W, r) * scale
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective, params0, args=(r,), method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
            
            if res.fun < 1e-8:
                U_opt = res.x[:num_U*r].reshape(num_U, r)
                V_opt = res.x[num_U*r:num_U*r+num_V*r].reshape(num_V, r)
                W_opt = res.x[num_U*r+num_V*r:].reshape(num_W, r)
                
                rounded = verify_and_round(U_opt, V_opt, W_opt, r)
                if rounded:
                    return rounded
                    
        return None

    start_time = time.time()
    
    # Try ranks that are likely to be minimal or close to it
    # 3x3x3 min rank is 23. 3x3x5 should be higher.
    # We try 30, 32, 35, 40, 44.
    ranks_to_try = [30, 32, 35, 40, 44]
    
    for r in ranks_to_try:
        if time.time() - start_time > 110:
            break
            
        result = try_rank(r, time_limit=15.0)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback: Schoolbook (Rank 45)
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
# EVOLVE-BLOCK-END
