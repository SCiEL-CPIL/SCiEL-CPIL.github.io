"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    # Build target tensor T
    # T[alpha, beta, gamma] = 1 if indices match
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
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
    schoolbook_rank = n * m * p

    # Fallback: Schoolbook solution
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
    best_rank = schoolbook_rank

    # Optimization search for lower rank
    # We only try a few ranks below schoolbook to avoid wasting time
    # if the rank is likely minimal.
    
    def make_residual_func(R):
        def res(params):
            len_U = N_U * R
            len_V = N_V * R
            U = params[:len_U].reshape(N_U, R)
            V = params[len_U:len_U+len_V].reshape(N_V, R)
            W = params[len_U+len_V:].reshape(N_W, R)
            
            # Compute Tensor Sum
            # T_calc[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            
            return (T_calc.flatten() - T_flat)
        
        return res

    def try_rank(R, time_limit):
        if R <= 0 or R >= best_rank:
            return None
        
        res_func = make_residual_func(R)
        total_vars = N_total * R
        
        best_sol = None
        best_cost = np.inf
        
        start_time = time.time()
        
        # Number of restarts is limited by time
        num_restarts = 5
        for _ in range(num_restarts):
            if time.time() > start_time + time_limit:
                break
            
            # Random initialization
            x0 = np.random.uniform(-1, 1, total_vars)
            try:
                sol = least_squares(res_func, x0, method='lm', max_nfev=500)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue
        
        if best_sol is None:
            return None
            
        # Check if error is close to 0
        if best_cost < 1e-8:
            # Round to half-integers
            x_rounded = np.round(best_sol * 2) / 2.0
            
            # Verify exactness with rounded solution
            res_rounded = res_func(x_rounded)
            if np.linalg.norm(res_rounded) < 1e-9:
                # Extract U, V, W
                len_U = N_U * R
                len_V = N_V * R
                U = x_rounded[:len_U].reshape(N_U, R)
                V = x_rounded[len_U:len_U+len_V].reshape(N_V, R)
                W = x_rounded[len_U+len_V:].reshape(N_W, R)
                
                # Convert to list of lists for output
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Deadline guard
    global_start = time.time()
    deadline = global_start + 10.0 # 10s budget for search, leaving plenty for safety
    
    # Try ranks from schoolbook-1 down to schoolbook-5
    # If the rank is truly lower, we might find it quickly.
    # If not, we stop and return schoolbook.
    for R in range(schoolbook_rank - 1, max(schoolbook_rank - 6, 1), -1):
        if time.time() > deadline:
            break
        
        result = try_rank(R, time_limit=1.5)
        if result is not None:
            best_result = result
            best_rank = R
            # Since we go downwards, the first success is the best (lowest) rank found in this loop
            # But we should continue to see if an even lower rank is possible?
            # Given the time constraints, finding one lower rank is a huge jump.
            # Let's break to save time for the next evaluation or just return.
            # Actually, let's continue if we have time, but 1.5s per rank is tight.
            # We'll break to ensure we return a valid result quickly.
            break
            
    return best_result
# EVOLVE-BLOCK-END
