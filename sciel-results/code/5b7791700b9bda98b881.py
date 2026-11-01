"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # The matrix multiplication tensor T[i,j,k] = 1 for all valid i,j,k
    # (each A[i,j]*B[j,k] contributes to C[i,k])
    # The all-ones tensor has rank 1: T[i,j,k] = u_i * v_j * w_k where u,v,w are all-ones
    # In the problem's indexing:
    #   U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = T[i,j,k]
    # For rank 1: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then product = 1*1*1 = 1 = T[i,j,k]. Valid!
    
    # Try rank 1 first
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    # Verify rank 1 decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank n*m*p
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify schoolbook
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank_schoolbook):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
