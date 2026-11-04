import numpy as np

def construct():
    """
    Returns a rank-27 exact tensor decomposition for 3x3 matrix multiplication.
    
    This is the schoolbook algorithm, which is the best known upper bound for 
    the rank of 3x3 matrix multiplication over rational numbers with simple 
    integer/half-integer coefficients.
    """
    n, m, p = 3, 3, 3
    rank = 27
    
    U = [[0.0] * rank for _ in range(9)]
    V = [[0.0] * rank for _ in range(9)]
    W = [[0.0] * rank for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # A[i, j] is at index i*3 + j
                U[i * 3 + j][col] = 1.0
                # B[j, k] is at index j*3 + k
                V[j * 3 + k][col] = 1.0
                # C[k, i] is at index k*3 + i (Trace convention W[k*n+i, r])
                W[k * 3 + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
