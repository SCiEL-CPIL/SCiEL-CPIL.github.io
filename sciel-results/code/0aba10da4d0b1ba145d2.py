"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, n*p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    return T

def _check_decomposition(U, V, W):
    """Check if U, V, W exactly decompose the tensor."""
    T = _build_tensor()
    # Compute sum_c U[:,c] * V[:,c] * W[:,c]
    r = U.shape[1]
    # Using einsum for efficient computation
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.array_equal(result, T)

def _to_list_of_lists(arr):
    """Convert numpy array to list of lists of float/int as required by interface."""
    return arr.tolist()

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
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

    def try_rank(R, time_limit, num_restarts=5):
        """
        Attempts to find a rank R decomposition using least squares optimization.
        Returns (U, V, W) as numpy arrays if successful, else None.
        """
        if R <= 0:
            return None
        
        res_func = make_residual_func(R)
        total_vars = N_total * R
        
        best_sol = None
        best_cost = np.inf
        
        start_time = time.time()
        
        # Number of restarts depends on time limit
        # We will try a few random restarts
        for restart in range(num_restarts):
            if time.time() > start_time + time_limit:
                break
            
            # Random initialization
            # Using small random values to help LM converge
            x0 = np.random.uniform(-1, 1, total_vars)
            
            try:
                # Levenberg-Marquardt is good for non-linear least squares
                sol = least_squares(res_func, x0, method='lm', max_nfev=1000)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
            except Exception:
                continue
        
        if best_sol is None:
            return None
            
        # Check if error is close to 0
        # If cost is small, try rounding to integers or half-integers
        # The problem allows integer/half-integer entries.
        if best_cost < 1e-4:
            # Try rounding to half-integers (multiply by 2, round, divide by 2)
            x_rounded = np.round(best_sol * 2) / 2.0
            
            # Verify exactness with rounded solution
            res_rounded = res_func(x_rounded)
            norm_res = np.linalg.norm(res_rounded)
            
            if norm_res < 1e-9:
                # Extract U, V, W
                len_U = N_U * R
                len_V = N_V * R
                U = x_rounded[:len_U].reshape(N_U, R)
                V = x_rounded[len_U:len_U+len_V].reshape(N_V, R)
                W = x_rounded[len_U+len_V:].reshape(N_W, R)
                
                return U, V, W
            
            # Also try rounding to integers just in case
            x_int = np.round(best_sol)
            res_int = res_func(x_int)
            norm_int = np.linalg.norm(res_int)
            if norm_int < 1e-9:
                len_U = N_U * R
                len_V = N_V * R
                U = x_int[:len_U].reshape(N_U, R)
                V = x_int[len_U:len_U+len_V].reshape(N_V, R)
                W = x_int[len_U+len_V:].reshape(N_W, R)
                return U, V, W

        return None

    # Deadline guard
    start_time = time.time()
    deadline = time.time() + 140.0 # 10s margin before 150s kill
    
    # First, ensure we have the schoolbook solution as a fallback
    schoolbook_rank = n * m * p
    U_school = np.zeros((N_U, schoolbook_rank))
    V_school = np.zeros((N_V, schoolbook_rank))
    W_school = np.zeros((N_W, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = schoolbook_rank
    
    # Try to find lower rank
    # Heuristic: try R from schoolbook-1 down to a reasonable limit
    # For 2x3x7, known bounds might be lower, but let's search downwards.
    # We limit the search to save time for optimization.
    min_rank_to_try = 20 # Arbitrary lower bound to avoid infinite loop if no solution found quickly
    
    for R in range(schoolbook_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        # Allocate time for this rank attempt
        # Decrease time limit as R increases? No, larger R is harder.
        # Let's give each rank attempt a fixed chunk of time, e.g., 5 seconds
        time_limit = 5.0
        
        result = try_rank(R, time_limit, num_restarts=3)
        
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = R
            # Since we go downwards, the first success is the best?
            # Yes, we want minimal rank.
            break
            
    return {"U": _to_list_of_lists(best_U), "V": _to_list_of_lists(best_V), "W": _to_list_of_lists(best_W)}
# EVOLVE-BLOCK-END
