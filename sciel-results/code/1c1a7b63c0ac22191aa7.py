"""
Program to find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 7 matrix.
The current best known simple decomposition is the schoolbook algorithm with rank 63.
"""
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    The shapes are (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the 1s for the current term
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
