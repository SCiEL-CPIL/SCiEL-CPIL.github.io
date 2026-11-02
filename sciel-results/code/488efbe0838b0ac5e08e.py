import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor M_{2,4,4}.
    Uses a pure Python combinatorial approach, iterating over indices explicitly
    to build the schoolbook decomposition. This is structurally distinct from
    vectorized NumPy approaches.
    """
    n, m, p = DIMENSIONS
    
    # Rank is n * m * p for the standard schoolbook decomposition
    rank = n * m * p
    
    # Initialize lists for U, V, W
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Calculate row indices
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                
                # Set the 1s
                U[u_row][c] = 1
                V[v_row][c] = 1
                W[w_row][c] = 1
                
                c += 1
                
    return {
        "U": U,
        "V": V,
        "W": W
    }
# EVOLVE-BLOCK-END
