"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n
    schoolbook_rank = n * m * p
    
    # Build the target tensor
    T = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Start with schoolbook solution
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
    best_rank = schoolbook_rank
    
    def verify_solution(U_arr, V_arr, W_arr):
        """Verify if the decomposition is exact."""
        R = U_arr.shape[1]
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_calc.flatten(), T_flat, atol=1e-9)
    
    def try_find_lower_rank(R, time_budget):
        """Try to find a valid decomposition of rank R."""
        if R >= best_rank:
            return None
        
        total_vars = (N_U + N_V + N_W) * R
        end_time = time.time() + time_budget
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > end_time:
                break
            
            # Random initialization
            x0 = np.random.uniform(-2, 2, total_vars)
            
            len_U = N_U * R
            len_V = N_V * R
            len_W = N_W * R
            
            try:
                # Use scipy if available, otherwise skip
                from scipy.optimize import least_squares
                
                def residuals(params):
                    U_p = params[:len_U].reshape(N_U, R)
                    V_p = params[len_U:len_U+len_V].reshape(N_V, R)
                    W_p = params[len_U+len_V:].reshape(N_W, R)
                    T_calc = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
                    return T_calc.flatten() - T_flat
                
                sol = least_squares(residuals, x0, method='lm', max_nfev=500)
                
                if sol.cost < 1e-6:
                    # Try rounding to half-integers
                    x_rounded = np.round(sol.x * 2) / 2.0
                    
                    U_r = x_rounded[:len_U].reshape(N_U, R)
                    V_r = x_rounded[len_U:len_U+len_V].reshape(N_V, R)
                    W_r = x_rounded[len_U+len_V:].reshape(N_W, R)
                    
                    if verify_solution(U_r, V_r, W_r):
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
            except Exception:
                continue
        
        return None
    
    # Try to find lower ranks - but given the problem structure, 
    # schoolbook is likely optimal. Quick check of a few ranks.
    for R in range(schoolbook_rank - 1, max(schoolbook_rank - 5, 30), -1):
        if time.time() > deadline - 10:
            break
        time_budget = min(10, deadline - time.time())
        result = try_find_lower_rank(R, time_budget)
        if result is not None:
            best_result = result
            best_rank = R
            break
    
    return best_result
# EVOLVE-BLOCK-END
