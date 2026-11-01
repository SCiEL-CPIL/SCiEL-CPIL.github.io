"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook decomposition for 2x2 * 2x14 multiplication.
    
    The tensor has indices:
      - Input A: n x m = 2 x 2 (indices i, j)
      - Input B: m x p = 2 x 14 (indices j, k)
      - Output C: n x p = 2 x 14 (indices i, k)
    
    The trace convention is W[k*n + i, r] where r is the rank index.
    
    For schoolbook:
      U[i*m + j, c] = 1 if c corresponds to (i,j,k)
      V[j*p + k, c] = 1 if c corresponds to (i,j,k)  
      W[k*n + i, c] = 1 if c corresponds to (i,j,k)
    
    Rank = n*m*p = 2*2*14 = 56
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
