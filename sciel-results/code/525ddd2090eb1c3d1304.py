import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-125 exact tensor decomposition for 5x5 x 5x5 matrix multiplication.
    Uses index decomposition from a flat rank index.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # r = i * m * p + j * p + k
        k = r % p
        tmp = r // p
        j = tmp % m
        i = tmp // m
        
        U[i * m + j][r] = 1
        V[j * p + k][r] = 1
        W[k * n + i][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
