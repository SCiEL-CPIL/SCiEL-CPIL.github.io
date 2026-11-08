"""Rewritten program for 6x6 matrix multiplication tensor decomposition."""
import time

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    with integer entries that exactly decompose the 6x6 matrix multiplication tensor.
    W follows the trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Use schoolbook decomposition: rank = n*m*p = 216
    # This is guaranteed correct and fast to construct.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
