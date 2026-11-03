"""Matrix multiplication 2x4x7 - tensor decomposition search."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 x 4x7 matrix multiplication.
    
    Uses the schoolbook construction with rank = n*m*p = 56.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # Schoolbook decomposition: rank = n * m * p
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
# EVOLVE-BLOCK-END
