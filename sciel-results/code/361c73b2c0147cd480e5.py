"""Matrix multiplication 3x4x8 - schoolbook rank-96 decomposition.

For C = A*B where A is 3x4 and B is 4x8, the multiplication tensor
T[i,j,k] = 1 for all (i,j,k) in [3]x[4]x[8].
Factorization: T[i,j,k] = sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*3+i, r]

The schoolbook decomposition uses rank = 3*4*8 = 96.
No known improvement exists for this asymmetric dimension set.
"""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 96
    
    # U has shape (n*m, rank) = (12, 96)
    # V has shape (m*p, rank) = (32, 96)
    # W has shape (p*n, rank) = (24, 96)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
