"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time
import random
import sys

DIMENSIONS = [4, 5, 5]
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    def _is_valid(U, V, W, r):
        """Check if the decomposition is exact for matrix multiplication."""
        if time.time() - start_time > TIME_LIMIT - 10:
            return False
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
        for i in range(n):
            if time.time() - start_time > TIME_LIMIT - 5:
                return False
            for j in range(m):
                for k in range(p):
                    lhs = 0.0
                    for c in range(r):
                        lhs += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(lhs - 1.0) > 1e-10:
                        return False
        return True

    def _pack(U, V, W, r):
        return U[:, :r].tolist(), V[:, :r].tolist(), W[:, :r].tolist()

    # Schoolbook decomposition as baseline
    r_school = n * m * p
    U_sb = np.zeros((n * m, r_school))
    V_sb = np.zeros((m * p, r_school))
    W_sb = np.zeros((p * n, r_school))
    
    indices = list(product(range(n), range(m), range(p)))
    for c, (i, j, k) in enumerate(indices):
        U_sb[i * m + j, c] = 1.0
        V_sb[j * p + k, c] = 1.0
        W_sb[k * n + i, c] = 1.0

    # Try to find a lower-rank decomposition using structured search
    # Strategy: Use the fact that we can share computations across output elements
    
    # For 4x5x5, let's try a decomposition with rank around 40-60
    # We'll use a greedy approach: try to assign terms that cover multiple (i,j,k) tuples
    
    best_result = (U_sb, V_sb, W_sb, r_school)
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 25
    # Try from 30 up to see if we can find something
    
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    rng = np.random.default_rng(42)
    
    # Try a few ranks with random search
    for r in [30, 35, 40, 45, 50, 55, 60]:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        for trial in range(20):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
                
            U = rng.choice(values, size=(n*m, r))
            V = rng.choice(values, size=(m*p, r))
            W = rng.choice(values, size=(p*n, r))
            
            if _is_valid(U, V, W, r):
                best_result = (U, V, W, r)
                break
        
        if best_result[3] < r_school:
            break
    
    # If we found a better result, use it; otherwise use schoolbook
    U_best, V_best, W_best, r_best = best_result
    U_list, V_list, W_list = _pack(U_best, V_best, W_best, r_best)
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
