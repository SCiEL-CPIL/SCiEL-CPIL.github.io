"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition as integer lists."""
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


def _verify(U, V, W, n, m, p, T_target):
    """
    Fast vectorized verification.
    """
    # U, V, W are numpy arrays of shape (dim, rank)
    # Compute T_calc[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    T_calc = np.einsum('ir,jr,kr->ijk', U, V, W)
    # Check exact match (using tolerance for floating point)
    return np.allclose(T_calc, T_target, atol=1e-9)


def _rank_from_lists(U, V, W, n, m, p):
    return len(U[0]) if U else 0


def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Precompute target tensor
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Candidate values for entries
    # Include half-integers to increase chance of finding a solution
    values = np.array([-2, -1, -0.5, 0, 0.5, 1, 2], dtype=float)
    
    # Try ranks from 15 upward, but prioritize 29 and 28
    # Since rank 30 is known, we try 29 first as it is theoretically possible
    ranks_to_try = [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15]
    
    for rank in ranks_to_try:
        if rank >= best_rank:
            continue
        if time.time() > deadline - 5:
            break
        
        # Adaptive number of attempts
        # More attempts for lower ranks? No, lower ranks are harder.
        # Start with reasonable attempts.
        max_attempts = 5000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random U, V, W with entries from 'values'
            U = values[np.random.randint(0, len(values), size=(n * m, rank))]
            V = values[np.random.randint(0, len(values), size=(m * p, rank))]
            W = values[np.random.randint(0, len(values), size=(p * n, rank))]
            
            # Check if this is a valid decomposition
            if _verify(U, V, W, n, m, p, T_target):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
        
        if best_rank == rank:
            break
    
    return best

# EVOLVE-BLOCK-END
