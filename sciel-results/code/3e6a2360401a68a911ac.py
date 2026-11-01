import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x3 by 3x3 matrix multiplication.
    This implementation uses the schoolbook algorithm, which yields a rank of n*m*p.
    For 2x3x3, this is rank 18.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices as lists of lists to avoid dependency on numpy for the final output structure
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map indices to linear indices
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u][col] = 1.0
                V[idx_v][col] = 1.0
                W[idx_w][col] = 1.0
                col += 1
                
    return {
        "U": U,
        "V": V,
        "W": W
    }
# EVOLVE-BLOCK-END
