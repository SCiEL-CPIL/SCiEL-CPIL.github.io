import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard matrix multiplication decomposition
    # C[i, j] = sum_k A[i, k] * B[k, j]
    # Each term (i, k, j) is a rank-1 component
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # U index: i * m + k
                u_idx = i * m + k
                # V index: k * p + j
                v_idx = k * p + j
                # W index: j * n + i (trace convention W[k*n+i, r] -> here j is col, i is row)
                # Task says W[k*n+i, r]. Here k is col of C, i is row of C.
                # So W index is j * n + i.
                w_idx = j * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
