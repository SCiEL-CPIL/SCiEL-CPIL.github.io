"""Matrix multiplication 3x3 by 3x6: rank-minimal tensor decomposition."""
import time

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Schoolbook: rank = n * m * p = 54
    # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    # Term r = i*m*p + j*p + k contributes to C[i,k] += A[i,j]*B[j,k]
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    
    rank = n * m * p
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
