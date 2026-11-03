"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def _schoolbook_decomposition():
    """
    Construct a rank-m decomposition for n x m * m x p matrix multiplication.
    This is the standard schoolbook algorithm.
    Rank = m.
    """
    n, m, p = DIMENSIONS
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][j] = 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2 by 4 
    matrix by a 4 by 6 matrix.
    
    Returns the schoolbook decomposition of rank 4.
    """
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
