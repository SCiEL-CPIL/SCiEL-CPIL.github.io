import numpy as np

def construct():
    n, m, p = 3, 3, 6
    rank = 54
    U = [[0] * rank for _ in range(n * m)]  # 9 x 54
    V = [[0] * rank for _ in range(m * p)]  # 18 x 54
    W = [[0] * rank for _ in range(p * n)]  # 18 x 54
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
