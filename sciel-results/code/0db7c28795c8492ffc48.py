import numpy as np

def construct():
    n, m, p = 4, 4, 5
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Part 1: k = 0..3 (M_{4,4,4} part)
    for k in range(4):
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Part 2: k = 4 (M_{4,4,1} part)
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
