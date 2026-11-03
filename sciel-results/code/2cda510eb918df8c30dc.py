import numpy as np
import time
import sys
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

    def make_residual_func(R):
        def res(params):
            # Unpack
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
        if R <= 0:
            return None
        
        res_func = make_residual_func(R)
        total_vars = N_total * R
        
        best_sol = None
        best_cost = np.inf
        
        start_time = time.time()
        
        # Number of restarts depends on time limit
        # Estimate time per restart. 
        # For small R, it's fast. For R ~ 40, it's slow.
        # Let's just run as many as we can.
        
        for restart in range(100): # Cap restarts to prevent infinite loop if time check fails
            if time.time() > start_time + time_limit:
                break
            
            x0 = np.random.uniform(-1, 1, total_vars)
            try:
                sol = least_squares(res_func, x0, method='lm', max_nfev=1000)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue
        
        if best_sol is None:
            return None
            
        # Check if error is close to 0
        if best_cost < 1e-4:
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
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Try to find minimal rank
    # Start from schoolbook rank
    schoolbook_rank = n * m * p
    best_result = None
    
    # First, ensure we have the schoolbook solution as a fallback
    U = [[0] * schoolbook_rank for _ in range(N_U)]
    V = [[0] * schoolbook_rank for _ in range(N_V)]
    W = [[0] * schoolbook_rank for _ in range(N_W)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank
    # Heuristic: try R from schoolbook-1 down to 10
    # Limit the number of tries to fit in time
    # Allocate time budget: 50% to first rank, 30% to second, etc.
    time_budget = deadline - (time.time() - start_time)
    
    ranks_to_try = list(range(schoolbook_rank - 1, max(5, schoolbook_rank - 5), -1))
    
    for idx, R in enumerate(ranks_to_try):
        if time.time() > deadline:
            break
        
        # Allocate decreasing time to lower ranks
        # e.g. 40s for R=41, 20s for R=40, 10s for R=39
        alloc_time = time_budget * (0.5 / (idx + 1)) 
        if alloc_time < 2: # Too little time
            break
            
        result = try_rank(R, alloc_time)
        if result is not None:
            best_result = result
            break
            
    return best_result
# EVOLVE-BLOCK-END
