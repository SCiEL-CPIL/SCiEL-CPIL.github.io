"""Program for matrix multiplication 2x3x7 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
import random
import sys
import itertools

DIMENSIONS = [2, 3, 7]

def _verify_float(U, V, W, n=2, m=3, p=7, tol=1e-8):
    """Verify the decomposition using floating point arithmetic."""
    rank = U.shape[1]
    # Pre-compute indices for speed
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            for k in range(p):
                idx_v = j * p + k
                idx_w = k * n + i
                # Compute sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                s = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n=2, m=3, p=7):
    """Verify with exact Fraction arithmetic."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            for k in range(p):
                idx_v = j * p + k
                idx_w = k * n + i
                s = Fraction(0)
                for r in range(rank):
                    # U, V, W are expected to be lists of floats that are exactly representable
                    # or we convert them to Fraction directly
                    u_val = Fraction(U[idx_u][r]).limit_denominator(1000)
                    v_val = Fraction(V[idx_v][r]).limit_denominator(1000)
                    w_val = Fraction(W[idx_w][r]).limit_denominator(1000)
                    s += u_val * v_val * w_val
                if s != 1:
                    return False
    return True

def _build_schoolbook():
    """Return the schoolbook rank-42 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _to_half_int_list(mat):
    """Convert numpy array to list of lists with half-integer precision."""
    return [[float(round(x * 2)) / 2.0 for x in row] for row in mat]

def _scipy_search(rank, n, m, p, deadline, seed=42):
    """
    Use scipy.optimize.least_squares to find a continuous decomposition.
    Then refine to half-integers.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Number of unknowns
    n_u = n * m
    n_v = m * p
    n_w = p * n
    total_vars = rank * (n_u + n_v + n_w)
    
    # Number of equations
    n_eq = n * m * p
    
    # Residual function
    def residuals(x):
        # Unpack
        U = x[:rank*n_u].reshape(n_u, rank)
        V = x[rank*n_u:rank*(n_u+n_v)].reshape(n_v, rank)
        W = x[rank*(n_u+n_v):].reshape(n_w, rank)
        
        # Compute residuals for each (i, j, k)
        res = np.zeros(n_eq)
        idx = 0
        for i in range(n):
            for j in range(m):
                idx_u = i * m + j
                for k in range(p):
                    idx_v = j * p + k
                    idx_w = k * n + i
                    res[idx] = 1.0 - np.dot(U[idx_u], V[idx_v] * W[idx_w])
                    idx += 1
        return res

    best_solution = None
    max_attempts = 5
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
            
        # Initialize x
        # Use a small random initialization
        x0 = np.random.randn(total_vars) * 0.5
        
        # Try to solve
        try:
            from scipy.optimize import least_squares
            # Bounds: keep values reasonable to prevent explosion
            bounds = (-5.0, 5.0)
            result = least_squares(residuals, x0, bounds=bounds, max_nfev=2000)
            
            if result.success or np.max(np.abs(result.fun)) < 1e-8:
                x_sol = result.x
                U = x_sol[:rank*n_u].reshape(n_u, rank)
                V = x_sol[rank*n_u:rank*(n_u+n_v)].reshape(n_v, rank)
                W = x_sol[rank*(n_u+n_v):].reshape(n_w, rank)
                
                # Check if it's a good continuous solution
                if np.max(np.abs(result.fun)) < 1e-6:
                    # Round to half-integers
                    U_list = _to_half_int_list(U)
                    V_list = _to_half_int_list(V)
                    W_list = _to_half_int_list(W)
                    
                    # Verify exact
                    if _verify_exact(U_list, V_list, W_list, n, m, p):
                        return U_list, V_list, W_list, rank
                    
                    # If exact verification fails, try local search on the rounded grid
                    # This is a simple hill-climbing to fix rounding errors
                    U_arr = np.array(U_list)
                    V_arr = np.array(V_list)
                    W_arr = np.array(W_list)
                    
                    improved = True
                    steps = 0
                    max_steps = 100
                    while improved and steps < max_steps and time.time() < deadline - 2:
                        improved = False
                        steps += 1
                        # Pick a random element to perturb
                        mat_choice = random.randint(0, 2)
                        if mat_choice == 0:
                            idx = random.randint(0, n_u - 1)
                            col = random.randint(0, rank - 1)
                            old_val = U_arr[idx, col]
                            # Try nearby half-integers
                            for delta in [-0.5, 0.5, -1.0, 1.0, -1.5, 1.5]:
                                new_val = old_val + delta
                                U_arr[idx, col] = new_val
                                if _verify_float(U_arr, V_arr, W_arr, n, m, p, tol=1e-8):
                                    improved = True
                                    break
                                U_arr[idx, col] = old_val
                        elif mat_choice == 1:
                            idx = random.randint(0, n_v - 1)
                            col = random.randint(0, rank - 1)
                            old_val = V_arr[idx, col]
                            for delta in [-0.5, 0.5, -1.0, 1.0, -1.5, 1.5]:
                                new_val = old_val + delta
                                V_arr[idx, col] = new_val
                                if _verify_float(U_arr, V_arr, W_arr, n, m, p, tol=1e-8):
                                    improved = True
                                    break
                                V_arr[idx, col] = old_val
                        else:
                            idx = random.randint(0, n_w - 1)
                            col = random.randint(0, rank - 1)
                            old_val = W_arr[idx, col]
                            for delta in [-0.5, 0.5, -1.0, 1.0, -1.5, 1.5]:
                                new_val = old_val + delta
                                W_arr[idx, col] = new_val
                                if _verify_float(U_arr, V_arr, W_arr, n, m, p, tol=1e-8):
                                    improved = True
                                    break
                                W_arr[idx, col] = old_val
                        
                        # Final check with exact arithmetic if improved
                        if improved:
                            U_list = _to_half_int_list(U_arr)
                            V_list = _to_half_int_list(V_arr)
                            W_list = _to_half_int_list(W_arr)
                            if _verify_exact(U_list, V_list, W_list, n, m, p):
                                return U_list, V_list, W_list, rank

        except Exception:
            pass
            
    return None

def construct():
    """Construct the tensor decomposition for 2x3x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Fallback: schoolbook
    schoolbook = _build_schoolbook()
    best_result = schoolbook
    best_rank = schoolbook_rank
    
    # Try ranks from lower bound up
    # Lower bound is 21. Known results for similar dimensions suggest rank is close to bound.
    ranks_to_try = [21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 35, 40, 41]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        if rank >= best_rank:
            continue
        
        # Try SciPy-based search
        result = _scipy_search(rank, n, m, p, deadline, seed=42 + rank)
        if result is not None:
            U, V, W, r = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break
        
    return best_result

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
