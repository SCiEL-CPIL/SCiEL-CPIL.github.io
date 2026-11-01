import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # The tensor for matrix multiplication (n x m) * (m x p) is a direct sum
    # of p copies of the (n, m, 1) tensor if we view it column-wise.
    # The rank of the (2, 3, 1) tensor is 6.
    # Thus the total rank is 9 * 6 = 54.
    # This is the same as the schoolbook rank n*m*p = 2*3*9 = 54.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strategy: Decompose into p independent (n, m, 1) tensor instances.
    # Each instance computes one column of the output matrix.
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # Index in U: row i*m + j
                U[i * m + j][col] = 1
                # Index in V: row j*p + k
                V[j * p + k][col] = 1
                # Index in W: row k*n + i (trace convention)
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
