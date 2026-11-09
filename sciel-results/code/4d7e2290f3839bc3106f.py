"""Program for matrix multiplication tensor decomposition 2x5x6."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=6
    
    # Build rank-1 terms for matrix multiplication tensor
    # (AB)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Each term: U[i*m+j] * V[j*p+k] * W[k*n+i]
    # Rank = n*m*p with unit entries
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
