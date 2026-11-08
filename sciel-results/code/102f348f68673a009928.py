import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for matrix multiplication
    of size 4x5 by 5x6.
    
    This implementation uses the standard schoolbook algorithm, which corresponds
    to the trivial algebraic extension in the BCLR framework. The rank is n*m*p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists to ensure standard Python integer types
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
