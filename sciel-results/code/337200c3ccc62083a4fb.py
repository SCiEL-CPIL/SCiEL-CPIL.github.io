"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

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
    
    def try_rank(R, time_budget, num_restarts=50):
        if R >= best_rank:
            return None
        
        res_func = make_residual_func(R)
        total_vars = (N_U + N_V + N_W) * R
        
        # Create structured initializations based on problem structure
        init_list = []
        
        # Strategy 1: Random uniform
        for _ in range(10):
            init_list.append(np.random.uniform(-1, 1, total_vars))
        
        # Strategy 2: Sparse random (many zeros)
        for _ in range(10):
            x = np.random.uniform(-1, 1, total_vars)
            mask = np.random.random(total_vars) < 0.4
            x[~mask] = 0
            init_list.append(x)
        
        # Strategy 3: Based on schoolbook structure with grouping
        # Group by j: for each j, try to cover all (i,k) pairs with fewer terms
        for grouping in [1, 2, 3]:
            x = np.zeros(total_vars)
            for j in range(m):
                # For this j, we need to cover i in {0,1}, k in {0,...,6}
                # Try to use 'grouping' terms per j
                for g in range(grouping):
                    r_idx = j * grouping + g
                    if r_idx >= R:
                        continue
                    # Assign which (i,k) pairs this term covers
                    for i in range(n):
                        for k in range(p):
                            if (i + k) % grouping == g:
                                # This term covers this (i,k) for this j
                                alpha = i * m + j
                                beta = j * p + k
                                gamma = k * n + i
                                # Set values to make the product contribute
                                # We need U[alpha,r]*V[beta,r]*W[gamma,r] = 1/(number of terms covering this)
                                x[alpha * R + r_idx] = 1.0
                                x[(N_U + beta) * R + r_idx] = 1.0
                                x[(N_U + N_V + gamma) * R + r_idx] = 1.0
            init_list.append(x)
        
        # Strategy 4: Use the fact that we can combine i=0 and i=1 for same (j,k)
        for scale in [0.5, 1.0, 1.5, 2.0]:
            x = np.zeros(total_vars)
            # For each (j,k), try to use one term that covers both i=0 and i=1
            # This requires W[k*n+0, r] and W[k*n+1, r] to be different
            for j in range(m):
                for k in range(p):
                    r_idx = j * p + k
                    if r_idx >= R:
                        continue
                    alpha0 = 0 * m + j  # i=0
                    alpha1 = 1 * m + j  # i=1
                    beta = j * p + k
                    gamma0 = k * n + 0
                    gamma1 = k * n + 1
                    # Set U, V, W so that:
                    # U[alpha0,r]*V[beta,r]*W[gamma0,r] = 1
                    # U[alpha1,r]*V[beta,r]*W[gamma1,r] = 1
                    # But also need cross terms to be 0, which is impossible with one term
                    # So this strategy won't work directly. Use small values instead.
                    x[alpha0 * R + r_idx] = scale
                    x[alpha1 * R + r_idx] = scale
                    x[(N_U + beta) * R + r_idx] = 1.0 / scale
                    x[(N_U + N_V + gamma0) * R + r_idx] = 1.0 / scale
                    x[(N_U + N_V + gamma1) * R + r_idx] = 1.0 / scale
            init_list.append(x)
        
        best_sol = None
        best_cost = np.inf
        
        end_time = time.time() + time_budget
        
        from scipy.optimize import least_squares
        
        for x0 in init_list:
            if time.time() > end_time:
                break
            try:
                sol = least_squares(res_func, x0, method='lm', max_nfev=500)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue
        
        # If we have time, try more random restarts
        while time.time() < end_time and best_cost > 1e-4:
            x0 = np.random.uniform(-2, 2, total_vars)
            try:
                sol = least_squares(res_func, x0, method='lm', max_nfev=300)
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
            ranks_to_try = [r for r in ranks_to_try if r < R]
    
    return best_result
# EVOLVE-BLOCK-END
