def construct():
    n, m, p = DIMENSIONS
    rank = 6
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # i=0, j=0
    U[0 * m + 0][0] = 1
    for k in range(p):
        V[0 * p + k][0] = 1
        W[k * n + 0][0] = 1
        
    # i=0, j=1
    U[0 * m + 1][1] = 1
    for k in range(p):
        V[1 * p + k][1] = 1
        W[k * n + 0][1] = 1
        
    # i=0, j=2
    U[0 * m + 2][2] = 1
    for k in range(p):
        V[2 * p + k][2] = 1
        W[k * n + 0][2] = 1
        
    # i=1, j=0
    U[1 * m + 0][3] = 1
    for k in range(p):
        V[0 * p + k][3] = 1
        W[k * n + 1][3] = 1
        
    # i=1, j=1
    U[1 * m + 1][4] = 1
    for k in range(p):
        V[1 * p + k][4] = 1
        W[k * n + 1][4] = 1
        
    # i=1, j=2
    U[1 * m + 2][5] = 1
    for k in range(p):
        V[2 * p + k][5] = 1
        W[k * n + 1][5] = 1
        
    return {"U": U, "V": V, "W": W}
