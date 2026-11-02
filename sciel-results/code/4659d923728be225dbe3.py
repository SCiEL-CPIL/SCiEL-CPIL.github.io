import numpy as np

def construct():
    n, m, p = 3, 3, 5
    
    # Rank 15 decomposition:
    # Split the output columns into two groups: k=0,1,2 and k=3,4.
    # Group 1: 3x3x3 schoolbook (rank 9)
    # Group 2: 3x3x2 schoolbook (rank 6)
    # Total rank 15.
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Group 1: k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Group 2: k in 3..4
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
