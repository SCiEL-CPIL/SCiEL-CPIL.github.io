def _decompose_1x2x7():
    rank = 14
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(14)]
    W = [[0]*rank for _ in range(7)]
    
    for j in range(2):
        for k in range(7):
            r = j*7 + k
            U[j][r] = 1
            V[j*7+k][r] = 1
            W[k][r] = 1
