"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

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
    deadline = 140.0  # Safety margin before 150s kill
    
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
        """Try to find a valid decomposition of rank R using gauge-fixed optimization."""
        if R >= best_rank:
            return None
        
        # Gauge fixing: fix W to have orthonormal columns (or use a simpler approach)
        # We'll fix U and V to be partially determined and optimize all three
        # but use a better initialization strategy
        
        total_vars = (N_U + N_V + N_W) * R
        end_time = time.time() + time_budget
        max_restarts = 50
        
        # Precompute indices for efficient residual computation
        U_flat_idx = np.arange(N_U * R)
        V_flat_idx = np.arange(N_U * R, (N_U + N_V) * R)
        W_flat_idx = np.arange((N_U + N_V) * R, total_vars)
        
        for restart in range(max_restarts):
            if time.time() > end_time:
                break
            
            # Strategy 1: Start from schoolbook-like structure with random perturbations
            if restart % 4 == 0:
                # Random uniform
                x0 = np.random.uniform(-1, 1, total_vars)
            elif restart % 4 == 1:
                # Sparse random - many zeros
                x0 = np.zeros(total_vars)
                num_nonzero = int(total_vars * 0.3)
                idx = np.random.choice(total_vars, num_nonzero, replace=False)
                x0[idx] = np.random.uniform(-1, 1, num_nonzero)
            elif restart % 4 == 2:
                # Binary-like initialization
                x0 = np.random.choice([0, 1, -1], total_vars)
            else:
                # Small Gaussian
                x0 = np.random.normal(0, 0.5, total_vars)
            
            len_U = N_U * R
            len_V = N_V * R
            len_W = N_W * R
            
            try:
                def residuals(params):
                    U_p = params[:len_U].reshape(N_U, R)
                    V_p = params[len_U:len_U+len_V].reshape(N_V, R)
                    W_p = params[len_U+len_V:].reshape(N_W, R)
                    T_calc = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
                    return T_calc.flatten() - T_flat
                
                def jac(params):
                    U_p = params[:len_U].reshape(N_U, R)
                    V_p = params[len_U:len_U+len_V].reshape(N_V, R)
                    W_p = params[len_U+len_V:].reshape(N_W, R)
                    # Jacobian is sparse in structure but we return dense for simplicity
                    # This is expensive, so we'll let least_squares estimate it
                    return None
                
                sol = least_squares(residuals, x0, method='lm', max_nfev=2000, xtol=1e-12, ftol=1e-12)
                
                if sol.cost < 1e-8:
                    # Try rounding to half-integers with multiple strategies
                    for round_factor in [2, 4, 8]:
                        x_rounded = np.round(sol.x * round_factor) / round_factor
                        
                        U_r = x_rounded[:len_U].reshape(N_U, R)
                        V_r = x_rounded[len_U:len_U+len_V].reshape(N_V, R)
                        W_r = x_rounded[len_U+len_V:].reshape(N_W, R)
                        
                        if verify_solution(U_r, V_r, W_r):
                            return {
                                "U": U_r.tolist(),
                                "V": V_r.tolist(),
                                "W": W_r.tolist()
                            }
                    
                    # Also try the raw solution
                    U_r = sol.x[:len_U].reshape(N_U, R)
                    V_r = sol.x[len_U:len_U+len_V].reshape(N_V, R)
                    W_r = sol.x[len_U+len_V:].reshape(N_W, R)
                    if verify_solution(U_r, V_r, W_r):
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
            except Exception:
                continue
        
        return None
    
    # Try to find lower ranks
    # The theoretical minimum rank for 2x3x7 is at least max(2*3, 3*7, 7*2) = 21
    # But likely higher. Let's try from schoolbook-1 down to a reasonable bound
    min_possible = max(n*m, m*p, p*n)
    search_start = schoolbook_rank - 1
    search_end = max(min_possible, 25)
    
    for R in range(search_start, search_end - 1, -1):
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time()
        if remaining < 2:
            break
        time_budget = min(8, remaining)
        result = try_find_lower_rank(R, time_budget)
        if result is not None:
            best_result = result
            best_rank = R
            break
    
    return best_result
# EVOLVE-BLOCK-END
