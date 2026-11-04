"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 145.0  # 150s limit with 5s safety margin

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _verify(U, V, W, n=4, m=5, p=5):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n=4, m=5, p=5):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Find a low-rank exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: shape (20, rank), V: shape (25, rank), W: shape (20, rank)
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Try to find a lower-rank decomposition
    # For now, use schoolbook as a safe fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try a structured search for lower rank
    # Given the time constraint, we'll try a few approaches
    
    # Approach 1: Try rank 99, 98, ... down to some minimum
    # This is expensive, so we'll only try a few ranks
    
    # For now, return schoolbook
    return best
# EVOLVE-BLOCK-END
