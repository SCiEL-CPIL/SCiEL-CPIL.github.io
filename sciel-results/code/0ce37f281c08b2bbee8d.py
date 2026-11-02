"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

import time
import itertools
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _check_exact(U, V, W, n, m, p):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _try_solve_for_W(U, V, n, m, p, r):
    """
    Given U and V, solve for W such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    Returns W if a valid integer/half-integer solution is found, else None.
    """
    num_vars = p * n * r
    num_eqs = n * m * p
    
    A = np.zeros((num_eqs, num_vars))
    b = np.ones(num_eqs)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    var_idx = c * (p * n) + (k * n + i)
                    A[eq_idx, var_idx] = U[i * m + j][c] * V[j * p + k][c]
                eq_idx += 1
    
    try:
        x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    
    # Check residuals
    if len(residuals) > 0 and residuals[0] > 1e-6:
        return None
        
    # Check if solution is close to integer/half-integer
    x_2x = 2 * x
    rounded_2x = np.round(x_2x)
    
    if np.allclose(x_2x, rounded_2x, atol=1e-6):
        W_flat = rounded_2x / 2.0
        W = [[0] * r for _ in range(p * n)]
        for c in range(r):
            for idx in range(p * n):
                W[idx][c] = W_flat[c * (p * n) + idx]
        
        if _check_exact(U, V, W, n, m, p):
            return W
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Lower bound
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound to best_rank - 1
    candidate_ranks = list(range(lower_bound, best_rank))
    
    random.seed(42)
    
    found = False
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
            
        # Limit attempts per rank
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize U and V with small integers
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            
            W = _try_solve_for_W(U, V, n, m, p, target_rank)
            if W is not None:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
