"""Matrix multiplication 2,3,7 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
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
    
    # Schoolbook solution
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
    deadline = 145.0  # 5s safety margin before 150s kill
    
    def verify_solution(U_arr, V_arr, W_arr, tol=1e-8):
        R = U_arr.shape[1]
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_calc.flatten(), T_flat, atol=tol)
    
    def try_rank_exact(R):
        """Try to find an exact rank-R decomposition using structured search."""
        if R >= best_rank:
            return None
        
        # For 2x3x7, we can exploit the structure.
        # Key insight: for each j (0..2), we need to decompose the 2x7 sub-problem.
        # The 2x7 matrix multiplication has a known structure.
        
        # Strategy: Use the fact that for n=2, we can sometimes combine terms.
        # Let's try a more systematic approach with better initializations.
        
        len_U = N_U * R
        len_V = N_V * R
        total_vars = (N_U + N_V + N_W) * R
        
        def residual(params):
            U = params[:len_U].reshape(N_U, R)
            V = params[len_U:len_U+len_V].reshape(N_V, R)
            W = params[len_U+len_V:].reshape(N_W, R)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return T_calc.flatten() - T_flat
        
        # Generate diverse initializations
        init_list = []
        
        # Init 1: Based on grouping by j
        # For each j, we have a 2x7 problem. Try to use R/m terms per j.
        terms_per_j = R // m
        if terms_per_j >= 2:
            x = np.zeros(total_vars)
            for j in range(m):
                for g in range(terms_per_j):
                    r_idx = j * terms_per_j + g
                    if r_idx >= R:
                        continue
                    # Assign (i,k) pairs to this term
                    for i in range(n):
                        for k in range(p):
                            if (i * p + k) % terms_per_j == g:
                                alpha = i * m + j
                                beta = j * p + k
                                gamma = k * n + i
                                x[alpha * R + r_idx] = 1.0
                                x[(N_U + beta) * R + r_idx] = 1.0
                                x[(N_U + N_V + gamma) * R + r_idx] = 1.0
            init_list.append(x)
        
        # Init 2: Random with structure
        for _ in range(20):
            x = np.random.uniform(-1, 1, total_vars)
            # Add some zeros to encourage sparsity
            mask = np.random.random(total_vars) < 0.3
            x[~mask] = 0
            init_list.append(x)
        
        # Init 3: Use known low-rank structures for 2xN problems
        # For 2x7, the border rank is known to be less than 14
        # Try to exploit this
        for scale in [0.5, 1.0, 2.0]:
            x = np.zeros(total_vars)
            for j in range(m):
                for k in range(p):
                    r_idx = j * p + k
                    if r_idx >= R:
                        continue
                    # Try to cover both i=0 and i=1 with one term
                    alpha0 = 0 * m + j
                    alpha1 = 1 * m + j
                    beta = j * p + k
                    gamma0 = k * n + 0
                    gamma1 = k * n + 1
                    x[alpha0 * R + r_idx] = scale
                    x[alpha1 * R + r_idx] = 1.0
                    x[(N_U + beta) * R + r_idx] = 1.0
                    x[(N_U + N_V + gamma0) * R + r_idx] = 1.0 / scale
                    x[(N_U + N_V + gamma1) * R + r_idx] = 1.0
            init_list.append(x)
        
        best_sol = None
        best_cost = np.inf
        
        end_time = time.time() + 30  # 30s per rank
        
        for x0 in init_list:
            if time.time() > end_time:
                break
            try:
                sol = least_squares(residual, x0, method='lm', max_nfev=1000)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
                    if best_cost < 1e-10:
                        break
            except Exception:
                continue
        
        # More random restarts if needed
        while time.time() < end_time and best_cost > 1e-6:
            x0 = np.random.uniform(-2, 2, total_vars)
            try:
                sol = least_squares(residual, x0, method='lm', max_nfev=500)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
                    if best_cost < 1e-10:
                        break
            except Exception:
                continue
        
        if best_sol is None or best_cost > 1e-4:
            return None
        
        # Try to round to half-integers
        x_rounded = np.round(best_sol * 2) / 2.0
        U_r = x_rounded[:len_U].reshape(N_U, R)
        V_r = x_rounded[len_U:len_U+len_V].reshape(N_V, R)
        W_r = x_rounded[len_U+len_V:].reshape(N_W, R)
        
        if verify_solution(U_r, W_r, V_r) or verify_solution(U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Try rounding to integers
        x_rounded_int = np.round(best_sol)
        U_r = x_rounded_int[:len_U].reshape(N_U, R)
        V_r = x_rounded_int[len_U:len_U+len_V].reshape(N_V, R)
        W_r = x_rounded_int[len_U+len_V:].reshape(N_W, R)
        
        if verify_solution(U_r, V_r, W_r):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # If very close, accept
        if best_cost < 1e-12:
            U_r = best_sol[:len_U].reshape(N_U, R)
            V_r = best_sol[len_U:len_U+len_V].reshape(N_V, R)
            W_r = best_sol[len_U+len_V:].reshape(N_W, R)
            if verify_solution(U_r, V_r, W_r, tol=1e-6):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        return None
    
    # Try ranks from 41 down to 20
    for R in range(schoolbook_rank - 1, 19, -1):
        if time.time() > deadline:
            break
        
        result = try_rank_exact(R)
        if result is not None:
            best_result = result
            best_rank = R
    
    return best_result
# EVOLVE-BLOCK-END
