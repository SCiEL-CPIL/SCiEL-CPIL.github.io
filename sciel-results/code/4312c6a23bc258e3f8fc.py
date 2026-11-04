"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The rank is n*m*p for this decoupled structure
    rank = n * m * p
    
    # Create zero matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Fill the matrices using vectorized operations or loops
    # We iterate over the "terms" of the decomposition
    # Each term corresponds to a unique (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                
                # U index: i*m + j
                U[i * m + j, col] = 1
                
                # V index: j*p + k
                V[j * p + k, col] = 1
                
                # W index: k*n + i
                W[k * n + i, col] = 1
                
    # Convert to lists of lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
