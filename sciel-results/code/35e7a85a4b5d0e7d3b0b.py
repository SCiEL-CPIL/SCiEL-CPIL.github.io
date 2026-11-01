import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Try rank 23 using Strassen for 2x2x2 subblock
    rank = 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2 (indices 0,1)
    # u, v, w are vectors of length 2
    strassen_u = [
        [1, 0], [0, 1], [1, 1], [1, 0], [0, 1], [1, 1], [1, -1]
    ]
    strassen_v = [
        [1, 0], [0, 1], [1, 0], [1, 1], [1, 1], [0, 1], [1, 1]
    ]
    strassen_w = [
        [1, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]
    ]
    
    # Fill first 7 columns
    for r in range(7):
        for i in range(2):
            U[i * m + 0 + 0][r] = strassen_u[r][i] # j=0
            U[i * m + 1][r] = strassen_u[r][i] # j=1
        for j in range(2):
            V[j * p + 0][r] = strassen_v[r][j] # k=0
            V[j * p + 1][r] = strassen_v[r][j] # k=1
        for k in range(2):
            W[k * n + 0][r] = strassen_w[r][k] # i=0
            W[k * n + 1][r] = strassen_w[r][k] # i=1

    # Fill remaining 16 columns with schoolbook for the rest
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this is in the 2x2x2 subblock
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
