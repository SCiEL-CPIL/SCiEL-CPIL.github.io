"""Program for rank-minimal tensor decomposition of 2x3x7 matrix multiplication."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import least_squares

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 by 3x7 matrix.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14
    schoolbook_rank = n * m * p  # 42
    
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
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = schoolbook_rank
    
    def verify_exact(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the tensor."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        result = np.einsum('rc,sc,tc->rst', U_arr, V_arr, W_arr)
        return np.max(np.abs(result - T)) < tol
    
    def try_rank_numeric(R, time_budget, max_restarts=30):
        """
        Try to find a valid decomposition of rank R using numerical optimization,
        then refine to exact half-integer entries.
        """
        if R >= best_rank:
            return None
        
        end_time = time.time() + time_budget
        total_vars = (N_U + N_V + N_W) * R
        
        def make_residual(R):
            def res(params):
                len_U = N_U * R
                len_V = N_V * R
                U = params[:len_U].reshape(N_U, R)
                V = params[len_U:len_U+len_V].reshape(N_V, R)
                W = params[len_U+len_V:].reshape(N_W, R)
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                return (T_calc.flatten() - T_flat)
            return res
        
        res_func = make_residual(R)
        best_cost = np.inf
        best_params = None
        
        for restart in range(max_restarts):
            if time.time() > end_time:
                break
            
            # Sparse random initialization: mostly zeros with some ±1 entries
            x0 = np.zeros(total_vars)
            # Add some random non-zero entries
            for _ in range(R * 3):
                idx = random.randint(0, total_vars - 1)
                x0[idx] = random.choice([-1, 1, -0.5, 0.5])
            
            try:
                sol = least_squares(res_func, x0, method='lm', max_nfev=2000)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_params = sol.x.copy()
                    # If we found an exact solution, return immediately
                    if best_cost < 1e-10:
                        break
            except Exception:
                continue
        
        if best_params is None or best_cost > 1e-4:
            return None
        
        # Now refine to exact half-integer entries
        # Round to nearest half-integer
        x_rounded = np.round(best_params * 2) / 2.0
        
        # Verify the rounded solution
        res_rounded = res_func(x_rounded)
        if np.linalg.norm(res_rounded) < 1e-9:
            len_U = N_U * R
            len_V = N_V * R
            U = x_rounded[:len_U].reshape(N_U, R)
            V = x_rounded[len_U:len_U+len_V].reshape(N_V, R)
            W = x_rounded[len_U+len_V:].reshape(N_W, R)
            
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Verify with the exact verification function
            if verify_exact(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try local refinement: perturb individual entries
        if time.time() < end_time:
            len_U = N_U * R
            len_V = N_V * R
            U = x_rounded[:len_U].reshape(N_U, R).copy()
            V = x_rounded[len_U:len_U+len_V].reshape(N_V, R).copy()
            W = x_rounded[len_U+len_V:].reshape(N_W, R).copy()
            
            # Try to improve by perturbing entries to nearby half-integers
            vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
            
            improved = True
            max_passes = 20
            for _pass in range(max_passes):
                if time.time() > end_time:
                    break
                if not improved:
                    break
                improved = False
                
                # Try perturbing U entries
                for row in range(N_U):
                    for c in range(R):
                        old = U[row, c]
                        for v in vals:
                            if abs(v - old) < 1e-9:
                                continue
                            U[row, c] = v
                            U_list = U.tolist()
                            V_list = V.tolist()
                            W_list = W.tolist()
                            if verify_exact(U_list, V_list, W_list):
                                return {"U": U_list, "V": V_list, "W": W_list}
                        U[row, c] = old
                
                # Try perturbing V entries
                for row in range(N_V):
                    for c in range(R):
                        old = V[row, c]
                        for v in vals:
                            if abs(v - old) < 1e-9:
                                continue
                            V[row, c] = v
                            U_list = U.tolist()
                            V_list = V.tolist()
                            W_list = W.tolist()
                            if verify_exact(U_list, V_list, W_list):
                                return {"U": U_list, "V": V_list, "W": W_list}
                        V[row, c] = old
                
                # Try perturbing W entries
                for row in range(N_W):
                    for c in range(R):
                        old = W[row, c]
                        for v in vals:
                            if abs(v - old) < 1e-9:
                                continue
                            W[row, c] = v
                            U_list = U.tolist()
                            V_list = V.tolist()
                            W_list = W.tolist()
                            if verify_exact(U_list, V_list, W_list):
                                return {"U": U_list, "V": V_list, "W": W_list}
                        W[row, c] = old
        
        return None
    
    # Try ranks from just below schoolbook down
    # Allocate time budget carefully
    time_elapsed = time.time() - start_time
    remaining = deadline - time_elapsed
    
    # Try a few ranks close to schoolbook first, then lower
    ranks_to_try = []
    for R in range(schoolbook_rank - 1, max(20, schoolbook_rank - 15), -1):
        ranks_to_try.append(R)
    
    for idx, R in enumerate(ranks_to_try):
        if time.time() > deadline:
            break
        
        # Allocate decreasing time to lower ranks
        # Give more time to higher ranks (closer to schoolbook) as they're more likely
        frac = 1.0 / (idx + 1)
        alloc_time = remaining * 0.8 * frac
        if alloc_time < 2.0:
            break
        
        result = try_rank_numeric(R, alloc_time, max_restarts=20)
        if result is not None:
            best_result = result
            best_rank = R
            break
    
    return best_result
# EVOLVE-BLOCK-END
