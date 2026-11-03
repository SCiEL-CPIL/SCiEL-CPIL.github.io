"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
START_TIME = time.time()
TIME_LIMIT = 120.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    """
    Searches for a low-rank exact tensor decomposition for 2x3x7 matrix multiplication
    using randomized local search with rational snapping.
    Falls back to schoolbook (rank 42) if search fails or times out.
    """
    n, m, p = DIMENSIONS
    
    # Try ranks from 41 down to 36
    # We allocate a time budget for each rank attempt
    start_time = time.time()
    time_limit = 110.0  # Leave 10s for safety
    
    # Ranks to try
    ranks_to_try = [41, 40, 39, 38, 37, 36]
    
    for r in ranks_to_try:
        if time.time() - start_time > time_limit:
            break
        
        # Allocate time for this rank (e.g., 10 seconds per rank)
        rank_time_limit = min(10.0, time_limit - (time.time() - start_time))
        
        result = _search_rank(n, m, p, r, rank_time_limit)
        if result is not None:
            return result
    
    # Fallback: schoolbook decomposition (rank 42)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            
    return {"U": U, "V": V, "W": W}

def _search_rank(n, m, p, rank, time_limit):
    """
    Attempts to find a decomposition of the given rank using randomized local search.
    """
    start_time = time.time()
    
    # Number of variables
    num_u = n * m * rank
    num_v = m * p * rank
    num_w = p * n * rank
    total_vars = num_u + num_v + num_w
    
    # Precompute indices for faster evaluation
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = (i * m + j) * rank
                v_idx = (j * p + k) * rank
                w_idx = (k * n + i) * rank
                indices.append((u_idx, v_idx, w_idx))
    
    def error_function(x):
        # Check time
        if time.time() - start_time > time_limit:
            raise TimeoutError("Time limit exceeded")
            
        U = x[:num_u].reshape((n * m, rank))
        V = x[num_u:num_u + num_v].reshape((m * p, rank))
        W = x[num_u + num_v:].reshape((p * n, rank))
        
        err = 0.0
        for u_idx, v_idx, w_idx in indices:
            # Sum over t
            s = 0.0
            for t in range(rank):
                s += U[u_idx // rank, u_idx % rank] * V[v_idx // rank, v_idx % rank] * W[w_idx // rank, w_idx % rank]
            err += (s - 1.0) ** 2
        return err
    
    # Try a few random restarts
    num_restarts = 5
    best_result = None
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        x0 = np.random.randn(total_vars) * 0.1
        # Scale to help convergence
        x0 = x0 * 2.0
        
        try:
            res = minimize(error_function, x0, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < 1e-6:
                # Extract U, V, W
                U_float = res.x[:num_u].reshape((n * m, rank))
                V_float = res.x[num_u:num_u + num_v].reshape((m * p, rank))
                W_float = res.x[num_u + num_v:].reshape((p * n, rank))
                
                # Attempt to snap to half-integers
                U_snap = np.round(U_float * 2) / 2
                V_snap = np.round(V_float * 2) / 2
                W_snap = np.round(W_float * 2) / 2
                
                # Verify exactness with snapped values
                if _verify_decomposition(n, m, p, U_snap, V_snap, W_snap, rank):
                    # Convert to list of Fractions for exact output
                    U_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in U_snap]
                    V_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in V_snap]
                    W_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in W_snap]
                    
                    return {"U": U_frac, "V": V_frac, "W": W_frac}
                
        except TimeoutError:
            break
        except Exception:
            continue
    
    return None

def _verify_decomposition(n, m, p, U, V, W, rank):
    """
    Verifies if the decomposition is exact.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-END
