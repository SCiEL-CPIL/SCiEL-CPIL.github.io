"""Matrix multiplication 3x3 by 3x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Vectorized construction of the Kronecker decomposition
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    for col in range(rank):
        # Decode col into (i, j, k)
        # col = i * m * p + j * p + k
        k = col % p
        j = (col // p) % m
        i = col // (m * p)
        
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
