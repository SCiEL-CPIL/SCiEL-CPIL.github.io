import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin before 150s kill

    # 1. Build the target tensor T
    # T[alpha, beta, gamma] = 1 if indices match
    # alpha = i*m + j (input A index)
    # beta = j*p + k (input B index)
    # gamma = k*n + i (output C index, trace convention W[k*n+i, r])
    T = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    T_flat = T.flatten()
    N_total = N_U + N_V + N_W

    def make_residual_func(R):
        def res(params):
            len_U = N_U * R
            len_V = N_V * R
            U = params[:len_U].reshape(N_U, R)
            V = params[len_U:len_U+len_V].reshape(N_V, R)
            W = params[len_U+len_V:].reshape(N_W, R)
            
            # Compute Tensor Sum: T_calc[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            
            return (T_calc.flatten() - T_flat)
        
        return res

    def try_rank(R, time_limit):
        if R <= 0:
            return None
        
        res_func = make_residual_func(R)
        total_vars = N_total * R
        
        best_sol = None
        best_cost = np.inf
        
        t_start = time.time()
        
        # Strategy: Use random restarts with Levenberg-Marquardt
        # We cap restarts to ensure we don't exceed time_limit
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > t_start + time_limit:
                break
            
            # Random initialization in [-1, 1]
            x0 = np.random.uniform(-1, 1, total_vars)
            
            try:
                # LM is robust for this size if R is close to optimal
                # max_nfev limited to prevent hanging on a single bad start
                sol = least_squares(res_func, x0, method='lm', max_nfev=500)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue
        
        if best_sol is None:
            return None
            
        # If we found a very low cost solution, attempt to round to half-integers
        # The problem allows integer/half-integer entries.
        if best_cost < 1e-5:
            # Round to nearest 0.5
            x_rounded = np.round(best_sol * 2) / 2.0
            
            # Verify exactness with rounded solution
            res_rounded = res_func(x_rounded)
            if np.linalg.norm(res_rounded) < 1e-9:
                len_U = N_U * R
                len_V = N_V * R
                U = x_rounded[:len_U].reshape(N_U, R)
                V = x_rounded[len_U:len_U+len_V].reshape(N_V, R)
                W = x_rounded[len_U+len_V:].reshape(N_W, R)
                
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # If rounding failed, maybe the solution is just integer?
            x_rounded_int = np.round(best_sol)
            res_rounded_int = res_func(x_rounded_int)
            if np.linalg.norm(res_rounded_int) < 1e-9:
                len_U = N_U * R
                len_V = N_V * R
                U = x_rounded_int[:len_U].reshape(N_U, R)
                V = x_rounded_int[len_U:len_U+len_V].reshape(N_V, R)
                W = x_rounded_int[len_U+len_V:].reshape(N_W, R)
                
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # 2. Construct Schoolbook fallback (Rank 42)
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(N_U)]
    V_school = [[0] * schoolbook_rank for _ in range(N_V)]
    W_school = [[0] * schoolbook_rank for _ in range(N_W)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # 3. Search for lower rank
    # Theoretical lower bound for matrix multiplication rank is often significantly lower than nmp.
    # For 2x3x7, known bounds might allow rank < 42.
    # We try ranks from 41 down to a reasonable limit.
    # We allocate more time to higher ranks (closer to schoolbook) as they are more likely to converge?
    # Actually, lower rank is harder. But we want the *minimal* rank.
    # So we must check 41, 40, 39... until we find one that works.
    
    # Heuristic: The search space is non-convex.
    # We will iterate R from schoolbook-1 down to 20.
    # We give each R a budget. If we find a valid decomposition for R, we return it immediately
    # (since we are going downwards, the first found is the minimal in our search range).
    
    # To maximize chance, we spend time proportional to the difficulty?
    # Let's just spend equal time chunks, stopping if time is up.
    
    search_start = schoolbook_rank - 1
    search_end = max(20, schoolbook_rank - 20) # Search down to 22
    
    for R in range(search_start, search_end - 1, -1):
        # Check time
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 2.0:
            break
            
        # Allocate a portion of remaining time to this rank
        # We have (search_start - R + 1) ranks left to try (including current)
        ranks_left = search_start - R + 1
        time_per_rank = remaining / ranks_left
        # Cap time per rank to avoid spending too long on one if others are easy
        time_per_rank = min(time_per_rank, 10.0) 
        
        result = try_rank(R, time_per_rank)
        
        if result is not None:
            best_result = result
            break # Found minimal rank in this range
            
    return best_result
# EVOLVE-BLOCK-END
