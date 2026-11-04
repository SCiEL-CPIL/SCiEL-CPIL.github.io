import time

N = 3
M = 4
P = 6

def construct():
    n, m, p = N, M, P
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Trivial decomposition: one rank-1 tensor for each (i, j, k)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: (i, j)
                U[i * m + j][r] = 1
                # V index: (j, k)
                V[j * p + k][r] = 1
                # W index: (k, i)  <-- Note: W[k*n+i, r]
                W[k * n + i][r] = 1
                r += 1

    return {'U': U, 'V': V, 'W': W}
