import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 12 decomposition
    rank = 12
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: i=0, k in {0, 1, 2}
    # Rank indices 0, 1, 2
    for j in range(m):
        U[0 * m + j][0] = 1
        U[0 * m + j][1] = 0
        U[0 * m + j][2] = 0
        # V for k=0,1,2
        for k in [0, 1, 2]:
            V[j * p + k][0] = 1
            V[j * p + k][1] = 0
            V[j * p + k][2] = 0
            
    W[0 * n + 0][0] = 1 # k=0, i=0 -> idx 0
    W[1 * n + 0][1] = 1 # k=1, i=0 -> idx 2
    W[2 * n + 0][2] = 1 # k=2, i=0 -> idx 4

    # Block 2: i=1, k in {0, 1, 2}
    # Rank indices 3, 4, 5
    for j in range(m):
        U[1 * m + j][3] = 1
        U[1 * m + j][4] = 0
        U[1 * m + j][5] = 0
        for k in [0, 1, 2]:
            V[j * p + k][3] = 1
            V[j * p + k][4] = 0
            V[j * p + k][5] = 0

    W[0 * n + 1][3] = 1 # k=0, i=1 -> idx 1
    W[1 * n + 1][4] = 1 # k=1, i=1 -> idx 3
    W[2 * n + 1][5] = 1 # k=2, i=1 -> idx 5

    # Block 3: i=0, k in {3, 4, 5}
    # Rank indices 6, 7, 8
    for j in range(m):
        U[0 * m + j][6] = 1
        U[0 * m + j][7] = 0
        U[0 * m + j][8] = 0
        for k in [3, 4, 5]:
            V[j * p + k][6] = 1
            V[j * p + k][7] = 0
            V[j * p + k][8] = 0

    W[3 * n + 0][6] = 1 # k=3, i=0 -> idx 6
    W[4 * n + 0][7] = 1 # k=4, i=0 -> idx 8
    W[5 * n + 0][8] = 1 # k=5, i=0 -> idx 10

    # Block 4: i=1, k in {3, 4, 5}
    # Rank indices 9, 10, 11
    for j in range(m):
        U[1 * m + j][9] = 1
        U[1 * m + j][10] = 0
        U[1 * m + j][11] = 0
        for k in [3, 4, 5]:
            V[j * p + k][9] = 1
            V[j * p + k][10] = 0
            V[j * p + k][11] = 0

    W[3 * n + 1][9] = 1 # k=3, i=1 -> idx 7
    W[4 * n + 1][10] = 1 # k=4, i=1 -> idx 9
    W[5 * n + 1][11] = 1 # k=5, i=1 -> idx 11

    return {"U": U, "V": V, "W": W}
