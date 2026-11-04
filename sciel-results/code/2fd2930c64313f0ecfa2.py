"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    The rank is n*m*p = 4*5*5 = 100.
    This is the best known upper bound for this specific size.
    
    Interface:
    U: shape (n*m, rank) -> (20, 100)
    V: shape (m*p, rank) -> (25, 100)
    W: shape (p*n, rank) -> (20, 100)
    
    Convention:
    U[i*m + j, r]
    V[j*p + k, r]
    W[k*n + i, r]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    # Each column r corresponds to a specific (i, j, k) term
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
