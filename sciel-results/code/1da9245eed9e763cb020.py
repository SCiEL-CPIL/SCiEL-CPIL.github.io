import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Direct computation: rank = n*m*p = 64
    # For each (i,j,k) in A[i,j]*B[j,k] -> C[i,k]
    # U[i*m+j, r] = 1 if r corresponds to (i,j,k), else 0
    # V[j*p+k, r] = 1 if r corresponds to (i,j,k), else 0
    # W[k*n+i, r] = 1 if r corresponds to (i,j,k), else 0
    
    rank = 64
    U = [[0] * rank for _ in range(n * m)]  # 8 x 64
    V = [[0] * rank for _ in range(m * p)]  # 32 x 64
    W = [[0] * rank for _ in range(p * n)]  # 16 x 64
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] * B[j,k] contributes to C[i,k]
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
