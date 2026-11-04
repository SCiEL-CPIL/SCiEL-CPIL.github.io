"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Try to find a low-rank decomposition
    # The condition: for all i,k, sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = delta[j1,j2] for all j1,j2
    
    # Start with schoolbook (rank 100) and try to reduce
    best_rank = n * m * p
    best_result = None
    
    # Try ranks from low to high
    for rank in range(1, best_rank + 1):
        if time.time() > deadline - 10:
            break
        if _try_rank(n, m, p, rank, deadline):
            best_rank = rank
            best_result = _get_decomposition(n, m, p, rank)
            if best_result is not None:
                break
    
    if best_result is None:
        # Fall back to schoolbook
        return _schoolbook(n, m, p)
    
    return best_result

def _schoolbook(n, m, p):
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

def _try_rank(n, m, p, rank, deadline):
    """Try to find a valid decomposition of given rank using random search."""
    # This is a placeholder; actual implementation would use optimization
    return False

def _get_decomposition(n, m, p, rank):
    """Get the decomposition found by _try_rank."""
    return None

# EVOLVE-BLOCK-END
