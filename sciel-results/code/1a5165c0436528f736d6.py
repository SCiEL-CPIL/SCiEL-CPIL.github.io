"""
Program for finding a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix.
The output factor follows the official trace convention W[k*n+i, r].
"""
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U has shape (n*m, rank) = (6, rank)
              V has shape (m*p, rank) = (15, rank)
              W has shape (p*n, rank) = (10, rank)
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition rank = n*m*p = 30
    # This is guaranteed to be correct
    rank = n * m * p
    
    # Use NumPy for efficient initialization and assignment
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
    
    # Convert to lists to match the original interface style
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
