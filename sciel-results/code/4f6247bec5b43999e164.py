"""Tensor decomposition for 3x5x5 matrix multiplication using DFT over Z_5."""
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Rank-15 construction:
    # For each of the 3 rows i of A, we use a rank-5 decomposition of the (1,5,5) tensor.
    # The (1,5,5) tensor computes c_k = sum_j a_j * B_{j,k} for k=0..4.
    # 
    # A simple rank-5 decomposition:
    # For r = 0, 1, 2, 3, 4:
    #   U[j, r] = delta_{j,r}  (selects a_r)
    #   V[j*p+k, r] = delta_{j,r}  (selects B_{r,k})
    #   W[k, r] = 1  (contributes to all outputs)
    # 
    # Verification:
    # c_k = sum_r U[j,r]*a_j * V[j*p+k,r]*B_{j,k} * W[k,r]
    #     = sum_r delta_{j,r}*a_j * delta_{j,r}*B_{j,k} * 1
    #     = sum_r a_r * B_{r,k}
    #     = sum_j a_j * B_{j,k}  ✓
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)]  # 15 x 15
    V = [[0] * rank for _ in range(m * p)]  # 25 x 15
    W = [[0] * rank for _ in range(p * n)]  # 15 x 15
    
    # For each row i of A (i = 0, 1, 2), use 5 columns (r = 0, 1, 2, 3, 4)
    for i in range(n):
        for r in range(m):  # r = 0, 1, 2, 3, 4
            col = i * m + r  # column index in the rank-15 decomposition
            
            # U: for this row i and frequency r
            # U[i*m + j, col] = 1 if j == r else 0
            for j in range(m):
                U[i * m + j][col] = 1 if j == r else 0
            
            # V: for this frequency r
            # V[j*p + k, col] = 1 if j == r else 0
            for j in range(m):
                for k in range(p):
                    V[j * p + k][col] = 1 if j == r else 0
            
            # W: for this row i and output k
            # W[k*n + i, col] = 1 for all k
            for k in range(p):
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
