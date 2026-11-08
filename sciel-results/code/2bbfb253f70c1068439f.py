import numpy as np

def construct():
    n, m, p = 2, 4, 7
    # Split m=4 into 4 parts of size 1.
    # Each part is a 2x1x7 multiplication.
    # Rank of 2x1x7 is 2.
    # Total rank is 4 * 2 = 8.
    
    rank = 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        # Part j corresponds to A[:, j] and B[j, :]
        # Term 1: A[0, j] * B[j, :] -> C[0, :]
        # L1(A) = A[0, j], M1(B) = B[j, k], W[0, k] = 1
        # U[0*m + j][col] = 1
        # V[j*p + k][col] = 1 for all k
        # W[k*n + 0][col] = 1 for all k
        U[0 * m + j][col] = 1
        for k in range(p):
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
        col += 1
        
        # Term 2: A[1, j] * B[j, :] -> C[1, :]
        # L2(A) = A[1, j], M2(B) = B[j, k], W[1, k] = 1
        # U[1*m + j][col] = 1
        # V[j*p + k][col] = 1 for all k
        # W[k*n + 1][col] = 1 for all k
        U[1 * m + j][col] = 1
        for k in range(p):
            V[j * p + k][col] = 1
            W[k * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
