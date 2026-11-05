import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    p1 = 3
    
    rank = 27 + 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    # Part 1: 3x3x3 (columns 0, 1, 2 of B)
    for i in range(n):
        for j in range(m):
            for k in range(p1):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Part 2: 3x3x4 (columns 3, 4, 5, 6 of B)
    for i in range(n):
        for j in range(m):
            for k in range(p1, p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
