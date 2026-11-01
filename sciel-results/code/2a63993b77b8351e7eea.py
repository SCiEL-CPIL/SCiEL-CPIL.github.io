"""Matrix multiplication 2x5x5 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find a low-rank exact decomposition for 2x5x5 matrix multiplication.
    Uses a rank-5 decomposition based on the structure of the identity matrix.
    """
    n, m, p = DIMENSIONS
    R = 5  # Rank
    
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    U = [[0] for _ in range(n * m)]
    V = [[0] for _ in range(m * p)]
    W = [[0] for _ in range(p * n)]
    
    # Initialize U
    # U[i*m + j, r] = 1 if j == r else 0
    for i in range(n):
        for j in range(m):
            for r in range(R):
                if j == r:
                    U[i * m + j][r] = 1
    
    # Initialize V
    # V[l*p + k, r] = 1 if l == r else 0
    for l in range(m):
        for k in range(p):
            for r in range(R):
                if l == r:
                    V[l * p + k][r] = 1
    
    # Initialize W
    # W[k*n + i, r] = 1 for all k, i, r
    for k in range(p):
        for i in range(n):
            for r in range(R):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
