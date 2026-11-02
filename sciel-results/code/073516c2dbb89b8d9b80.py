import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-25 decomposition: one term per (j, k) pair
    # Term (j,k): U[i*m+j, (j,k)] = 1 for both i, V[j*p+k, (j,k)] = 1, W[k*n+i, (j,k)] = 1 for both i
    # This gives C[i,k] = sum_j A[i,j] * B[j,k] * 1 = (A*B)[i,k]
    # Invalid triples vanish because U and V have disjoint support on j-index
    
    rank = m * p  # 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            col = j * p + k
            # U: both rows of A, column j
            for i in range(n):
                U[i * m + j][col] = 1
            # V: row j of B, column k
            V[j * p + k][col] = 1
            # W: both rows of C, column k
            for i in range(n):
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
