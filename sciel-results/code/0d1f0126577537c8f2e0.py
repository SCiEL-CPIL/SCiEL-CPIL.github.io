"""
Optimized construction for Matrix Multiplication Tensor Decomposition.
Task: Find rank-minimal exact tensor decomposition for 2x3 * 3x7.
Dimensions: n=2, m=3, p=7.
Schoolbook rank: 42.
"""

import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def _build_target_tensor(n, m, p):
    """Construct the flattened target tensor for matrix multiplication."""
    # Shape: (n*m, m*p, p*n)
    # Index u = i*m + j
    # Index v = j*p + k
    # Index w = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _residual_func(params, r, n, m, p, T_flat):
    """
    Compute residual for a given rank r and parameters.
    params: flattened U, V, W
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    """
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    U = params[:size_u * r].reshape((size_u, r))
    V = params[size_u * r : size_u * r + size_v * r].reshape((size_v, r))
    W = params[size_u * r + size_v * r :].reshape((size_w, r))
    
    # Compute tensor M = sum_k U[:,k] o V[:,k] o W[:,k]
    # Using einsum: 'ti,tj,tk->ijk'
    # A = U.T (r x n*m)
    # B = V.T (r x m*p)
    # C = W.T (r x p*n)
    A = U.T
    B = V.T
    C = W.T
    
    M = np.einsum('ti,tj,tk->ijk', A, B, C)
    
    # Residual
    R = M - T_flat.reshape((n * m, m * p, p * n))
    return R.flatten()

def _round_to_half_integers(matrix):
    """Round values to the nearest half-integer (0, 0.5, 1, 1.5, etc.)."""
    # Multiply by 2, round to nearest integer, divide by 2
    return np.round(matrix * 2) / 2

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for 120s limit
    
    # Precompute target tensor
    T = _build_target_tensor(n, m, p)
    T_flat = T.flatten()
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1.0
                V_sb[j * p + k, c] = 1.0
                W_sb[k * n + i, c] = 1.0
                c += 1
                
    best_rank = rank_schoolbook
    best_solution = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    
    # Try to find a lower rank
    # We try ranks from 41 down to 35 (or until time runs out)
    # Theoretical lower bound is max(nm, mp, pn) = 21, but 41-35 is a reasonable search space
    # for potential Strassen-like improvements.
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    total_vars = size_u + size_v + size_w
    
    for r in range(rank_schoolbook - 1, 34, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Multiple restarts for each rank to improve chance of finding a solution
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Different random seeds for each restart
            seed = 42 + r * 100 + restart
            np.random.seed(seed)
            
            x0 = np.random.randn(total_vars * r)
            
            try:
                result = least_squares(
                    _residual_func,
                    x0,
                    args=(r, n, m, p, T_flat),
                    method='trf',  # Trust Region Reflective
                    max_nfev=1000,  # Limit function evaluations
                    xtol=1e-8,
                    ftol=1e-8,
                    gtol=1e-8
                )
            except Exception as e:
                continue
                
            if result.success or result.cost < 1e-6:
                # Check if the solution is valid
                # Round to half-integers
                params_opt = result.x
                U_opt = params_opt[:size_u * r].reshape((size_u, r))
                V_opt = params_opt[size_u * r : size_u * r + size_v * r].reshape((size_v, r))
                W_opt = params_opt[size_u * r + size_v * r :].reshape((size_w, r))
                
                U_rounded = _round_to_half_integers(U_opt)
                V_rounded = _round_to_half_integers(V_opt)
                W_rounded = _round_to_half_integers(W_opt)
                
                # Verify the rounded solution
                # Reconstruct tensor
                A = U_rounded.T
                B = V_rounded.T
                C = W_rounded.T
                M_check = np.einsum('ti,tj,tk->ijk', A, B, C)
                
                residual_check = np.linalg.norm(M_check - T)
                
                if residual_check < 1e-9:
                    best_rank = r
                    best_solution = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    # Found a valid solution at this rank, break out of restart loop
                    break
                else:
                    # Rounded solution is invalid. Check if unrounded is "close enough" for the evaluator?
                    # The prompt says "exactly decompose". So we need exact integers/half-integers.
                    # If rounding fails, this rank is not a solution.
                    pass
    return best_solution
