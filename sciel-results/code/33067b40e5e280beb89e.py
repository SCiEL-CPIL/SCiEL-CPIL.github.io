"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Schoolbook decomposition for 2x2 * 2x14 multiplication.
    
    Rank = n*m*p = 2*2*14 = 56
    
    U[i*m + j, r] = 1 if r corresponds to (i,j,k)
    V[j*p + k, r] = 1 if r corresponds to (i,j,k)
    W[k*n + i, r] = 1 if r corresponds to (i,j,k)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1.0
                V[j * p + k][r] = 1.0
                W[k * n + i][r] = 1.0
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
