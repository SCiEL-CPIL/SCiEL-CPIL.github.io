"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Matrix multiplication 3x3 by 3x6.
    Returns a schoolbook decomposition with rank 54.
    
    U: (n*m, rank) = (9, 54)
    V: (m*p, rank) = (18, 54)
    W: (p*n, rank) = (18, 54)
    
    W[k*n+i, r] follows the official trace convention.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
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
                
    # Convert to lists for the required interface if necessary, 
    # but numpy arrays are usually accepted or can be converted.
    # The task asks for factor matrices.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
