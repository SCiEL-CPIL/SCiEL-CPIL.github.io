import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a sum of m=4 rank-1 tensors:
    # For each j: u[i,j'] = delta(j',j), v[j',k] = delta(j',j), w[i,k] = 1
    # Rank = m = 4
    
    rank = m  # = 4
    
    U = [[0] * rank for _ in range(n * m)]  # (n*m, rank) = (8, 4)
    V = [[0] * rank for _ in range(m * p)]  # (m*p, rank) = (16, 4)
    W = [[0] * rank for _ in range(p * n)]  # (p*n, rank) = (8, 4)
    
    for j in range(m):
        # U: pick out A[i,j] for all i
        for i in range(n):
            U[i * m + j][j] = 1
        # V: pick out B[j,k] for all k
        for k in range(p):
            V[j * p + k][j] = 1
        # W: full contribution to C[i,k]
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
