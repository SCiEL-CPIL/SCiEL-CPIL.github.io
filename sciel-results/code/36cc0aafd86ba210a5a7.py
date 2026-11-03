def construct():
    n, m, p = 2, 3, 8
    rank = 6
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # k=0
    # r=0, 1
    # U[i, 0, r]
    U[0 * m + 0][0] = 1 # i=0, k=0, r=0
    U[1 * m + 0][1] = 1 # i=1, k=0, r=1
    # V[0, j, r]
    for j in range(p):
        V[0 * p + j][0] = 1
        V[0 * p + j][1] = 1
    # W[j, i, r]
    for j in range(p):
        W[j * n + 0][0] = 1 # i=0
        W[j * n + 1][1] = 1 # i=1
        
    # k=1
    # r=2, 3
    U[0 * m + 1][2] = 1
    U[1 * m + 1][3] = 1
    for j in range(p):
        V[1 * p + j][2] = 1
        V[1 * p + j][3] = 1
    for j in range(p):
        W[j * n + 0][2] = 1
        W[j * n + 1][3] = 1
        
    # k=2
    # r=4, 5
    U[0 * m + 2][4] = 1
    U[1 * m + 2][5] = 1
    for j in range(p):
        V[2 * p + j][4] = 1
        V[2 * p + j][5] = 1
    for j in range(p):
        W[j * n + 0][4] = 1
        W[j * n + 1][5] = 1
        
    return {"U": U, "V": V, "W": W}
