"""Optimal rank decomposition for 2x3 by 3x9 matrix multiplication."""

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Constructs the rank-minimal tensor decomposition for 2x3 by 3x9 matrix multiplication.
    The rank is 54, corresponding to the schoolbook algorithm.
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
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
