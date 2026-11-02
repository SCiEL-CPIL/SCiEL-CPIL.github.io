"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a bilinear decomposition for (2,5,5) matrix multiplication.
    Uses vectorized NumPy construction for the schoolbook decomposition.
    """
    import numpy as np
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    rank = N * M // m * p # n*m*p
    rank = n * m * p
    
    U = np.zeros((N, rank))
    V = np.zeros((M, rank))
    W = np.zeros((P, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
