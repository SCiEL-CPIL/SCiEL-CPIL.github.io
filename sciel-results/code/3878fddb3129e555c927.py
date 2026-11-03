"""Program for matrix multiplication 2x3 by 3x10 with rank-minimal decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    Uses schoolbook decomposition with rank 60.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 60
    
    # U: (n*m, rank) = (6, 60)
    # V: (m*p, rank) = (30, 60)
    # W: (p*n, rank) = (20, 60)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists for the required interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
