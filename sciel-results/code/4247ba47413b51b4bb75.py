"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a valid exact tensor decomposition for multiplying 
    a 3x3 matrix by a 3x8 matrix using the trivial decomposition.
    
    The trivial rank is n * m * p.
    For 3x3 * 3x8, rank = 3 * 3 * 8 = 72.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists to ensure compatibility with evaluator
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Trivial decomposition: one term for each (i, l, k)
    # r = (i * m + l) * p + k
    for i in range(n):
        for l in range(m):
            for k in range(p):
                r = (i * m + l) * p + k
                
                # U index: i*m + l
                U[i * m + l][r] = 1
                
                # V index: l*p + k
                V[l * p + k][r] = 1
                
                # W index: k*n + i
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
