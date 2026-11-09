import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    start_time = time.time()
    time_limit = 140  # 10s margin

    n_sub, m_sub, p_sub = 2, 3, 3
    
    # Tensor T_sub of shape (n_sub*m_sub, m_sub*p_sub, p_sub*n_sub)
    T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub), dtype=float)
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                T_sub[i * m_sub + j, j * p_sub + k, k * n_sub + i] = 1.0

    def try_find_decomposition(r, max_time=20):
        if time.time() - start_time > time_limit:
            return None
            
        # Random initialization with half-integers
        # Values in {-1.5, -1, -0.5, 0, 0.5, 1, 1.5}
        vals = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        
        def objective(params):
            U_flat = params[:n_sub * m_sub * r].reshape(n_sub * m_sub, r)
            V_flat = params[n_sub * m_sub * r : n_sub * m_sub * r + m_sub * p_sub * r].reshape(m_sub * p_sub, r)
            W_flat = params[n_sub * m_sub * r + m_sub * p_sub * r :].reshape(p_sub * n_sub, r)
            
            T_comp = np.einsum('it,jt,kt->ijk', U_flat, V_flat, W_flat)
            diff = T_comp - T_sub
            return np.sum(diff * diff)
        
        best_res = None
        best_val = float('inf')
        
        # Try multiple restarts
        num_restarts = 50
        for _ in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize with random half-integers
            U = np.random.choice(vals, size=(n_sub * m_sub, r))
            V = np.random.choice(vals, size=(m_sub * p_sub, r))
            W = np.random.choice(vals, size=(p_sub * n_sub, r))
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead for derivative-free optimization
            try:
                res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
                if res.fun < best_val:
                    best_val = res.fun
                    best_res = res
            except Exception:
                pass
        
        if best_res and best_res.fun < 1e-6:
            U_opt = best_res.x[:n_sub * m_sub * r].reshape(n_sub * m_sub, r)
            V_opt = best_res.x[n_sub * m_sub * r : n_sub * m_sub * r + m_sub * p_sub * r].reshape(m_sub * p_sub, r)
            W_opt = best_res.x[n_sub * m_sub * r + m_sub * p_sub * r :].reshape(p_sub * n_sub, r)
            
            # Snap to nearest half-integer
            U_int = np.round(U_opt * 2) / 2
            V_int = np.round(V_opt * 2) / 2
            W_int = np.round(W_opt * 2) / 2
            
            # Verify
            T_check = np.einsum('it,jt,kt->ijk', U_int, V_int, W_int)
            if np.allclose(T_check, T_sub, atol=1e-9):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
        return None

    # Try ranks from 15 upwards
    for r in [15, 16]:
        if time.time() - start_time > time_limit:
            break
        res = try_find_decomposition(r)
        if res:
            U_sub, V_sub, W_sub = res
            global_rank = 2 * r
            U_global = [[0] * global_rank for _ in range(n * m)]
            V_global = [[0] * global_rank for _ in range(m * p)]
            W_global = [[0] * global_rank for _ in range(p * n)]
            
            for t in range(r):
                for i in range(n * m):
                    U_global[i][t] = U_sub[i][t]
                    U_global[i][t + r] = U_sub[i][t]
                    
            for t in range(r):
                for j in range(m_sub * p_sub):
                    V_global[j][t] = V_sub[j][t]
                    V_global[j + m_sub * p_sub][t + r] = V_sub[j][t]
                    
            for t in range(r):
                for k in range(p_sub * n_sub):
                    W_global[k][t] = W_sub[k][t]
                    W_global[k + p_sub * n_sub][t + r] = W_sub[k][t]
                    
            return {"U": U_global, "V": V_global, "W": W_global}
            
    return {"U": U_school, "V": V_school, "W": W_school}
