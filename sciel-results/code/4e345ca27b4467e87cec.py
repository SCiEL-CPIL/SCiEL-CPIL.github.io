"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14
    
    # Build target tensor T
    T = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
    T_flat = T.flatten()
    
    # Schoolbook solution as fallback
    schoolbook_rank = n * m * p  # 42
    U_sb = [[0] * schoolbook_rank for _ in range(N_U)]
    V_sb = [[0] * schoolbook_rank for _ in range(N_V)]
    W_sb = [[0] * schoolbook_rank for _ in range(N_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    def verify_solution(U_arr, V_arr, W_arr):
        R = U_arr.shape[1]
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_calc.flatten(), T_flat, atol=1e-8)
    
    def make_residual_func(R):
        len_U = N_U * R
        len_V = N_V * R
        def res(params):
            U = params[:len_U].reshape(N_U, R)
            V = params[len_U:len_U+len_V].reshape(N_V, R)
            W = params[len_U+len_V:].reshape(N_W, R)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return T_calc.flatten() - T_flat
        return res
    
    def try_rank(R, time_budget, num_restarts=30):
        if R >= best_rank:
            return None
        
        res_func = make_residual_func(R)
        total_vars = (N_U + N_V + N_W) * R
        
        # Create structured initializations
        init_list = []
        
        # 1. Random uniform
        for _ in range(5):
            init_list.append(np.random.uniform(-1, 1, total_vars))
        
        # 2. Small random
        for _ in range(5):
            init_list.append(np.random.uniform(-0.5, 0.5, total_vars))
        
        # 3. Sparse random (many zeros)
        for _ in range(5):
            x = np.random.uniform(-1, 1, total_vars)
            mask = np.random.random(total_vars) < 0.3
            x[~mask] = 0
            init_list.append(x)
        
        # 4. Based on schoolbook structure (scaled)
        for scale in [0.5, 1.0, 2.0]:
            x = np.zeros(total_vars)
            # Fill with small values based on schoolbook pattern
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if i * m + j < N_U:
                            idx_u = (i * m + j) * R + (j * p + k) % R
                            x[idx_u] = scale * 0.1
                        if j * p + k < N_V:
                            idx_v = (j * p + k) * R + (i * p + k) % R
                            x[idx_v] = scale * 0.1
                        if k * n + i < N_W:
                            idx_w = (k * n + i) * R + (i * m + j) % R
                            x[idx_w] = scale * 0.1
            init_list.append(x)
        
        best_sol = None
        best_cost = np.inf
        
        end_time = time.time() + time_budget
        
        for x0 in init_list:
            if time.time() > end_time:
                break
            try:
                sol = least_squares(res_func, x0, method='lm', max_nfev=1000)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue
        
        # If we have time, try more random restarts
        while time.time() < end_time and best_cost > 1e-4:
            x0 = np.random.uniform(-2, 2, total_vars)
            try:
                sol = least_squares(res_func, x0, method='lm', max_nfev=500)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue
        
        if best_sol is None or best_cost > 1e-3:
            return None
        
        # Try rounding to half-integers
        x_rounded = np.round(best_sol * 2) / 2.0
        
        U_r = x_rounded[:N_U * R].reshape(N_U, R)
        V_r = x_rounded[N_U * R:N_U * R + N_V * R].reshape(N_V, R)
        W_r = x_rounded[N_U * R + N_V * R:].reshape(N_W, R)
        
        if verify_solution(U_r, V_r, W_r):
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
        
        # Try rounding to integers
        x_rounded_int = np.round(best_sol)
        U_r = x_rounded_int[:N_U * R].reshape(N_U, R)
        V_r = x_rounded_int[N_U * R:N_U * R + N_V * R].reshape(N_V, R)
        W_r = x_rounded_int[N_U * R + N_V * R:].reshape(N_W, R)
        
        if verify_solution(U_r, V_r, W_r):
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
        
        # Try the unrounded solution if it's very close
        if best_cost < 1e-10:
            U_r = best_sol[:N_U * R].reshape(N_U, R)
            V_r = best_sol[N_U * R:N_U * R + N_V * R].reshape(N_V, R)
            W_r = best_sol[N_U * R + N_V * R:].reshape(N_W, R)
            if verify_solution(U_r, V_r, W_r):
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
        return None
    
    # Try ranks from schoolbook-1 down to 25
    # Focus on the most promising range first
    ranks_to_try = list(range(schoolbook_rank - 1, 24, -1))
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        # Allocate time: more time for higher ranks (closer to schoolbook)
        if R > 35:
            time_budget = min(20, remaining * 0.5)
        elif R > 30:
            time_budget = min(25, remaining * 0.4)
        else:
            time_budget = min(30, remaining * 0.3)
        
        result = try_rank(R, time_budget)
        if result is not None:
            best_result = result
            best_rank = R
            # Continue searching for even lower ranks
            # But reduce the search space
            ranks_to_try = [r for r in ranks_to_try if r < R]
    
    return best_result
# EVOLVE-BLOCK-END
