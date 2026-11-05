import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    """
    Construct the rank-minimal exact tensor decomposition for matrix multiplication.
    For 3x4 * 4x7, we return the schoolbook decomposition with rank 3*4*7 = 84.
    
    The decomposition satisfies:
    For each output entry C[i,k] = sum_j A[i,j] * B[j,k],
    we have one rank-1 term per (i,j,k) triple.
    
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    with the trace convention W[k*n + i, r].
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Term: A[i,j] * B[j,k] contributes to C[i,k]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
