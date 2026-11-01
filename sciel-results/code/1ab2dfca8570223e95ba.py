import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Tensor slicing: split middle dimension into two parts
    # A = [A_L | A_R] where A_L is 2x2, A_R is 2x2
    # B = [B_L; B_R] where B_L is 2x7, B_R is 2x7
    # C = A_L @ B_L + A_R @ B_R
    # Each product is 2x2x7, rank 14 with schoolbook
    # Total rank = 14 + 14 = 28
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 28 x 28
    W = [[0] * rank for _ in range(p * n)]  # 14 x 28
    
    column = 0
    # Left part: j=0,1
    for i in range(n):
        for j in range(2):  # j=0,1
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Right part: j=2,3
    for i in range(n):
        for j in range(2, 4):  # j=2,3
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
