import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    total_terms = n * m * p
    U = [[0.0] * total_terms for _ in range(n * m)]
    V = [[0.0] * total_terms for _ in range(m * p)]
    W = [[0.0] * total_terms for _ in range(p * n)]
    
    counter = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][counter] = 1.0
                V[j * p + k][counter] = 1.0
                W[k * n + i][counter] = 1.0
                counter += 1
    
    return {"U": U, "V": V, "W": W}
