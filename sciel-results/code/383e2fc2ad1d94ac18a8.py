def construct():
    n, m, p = 2, 4, 7
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: j in {0, 1}
    # c indexes 0 to 13
    c = 0
    for i in range(n):
        for j in range(2): # j=0, 1
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
                
    # Part 2: j in {2, 3}
    # c indexes 14 to 27
    for i in range(n):
        for j in range(2, 4): # j=2, 3
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}
