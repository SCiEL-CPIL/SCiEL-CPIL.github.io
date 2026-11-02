"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for 2x3 matrix multiplication by 3x9.
    Rank is n*m*p = 2*3*9 = 54.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Fill in the schoolbook terms
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Convert to lists of lists as per interface requirement
    # The interface expects {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # and integer/half-integer entries.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
