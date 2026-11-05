import numpy as np

def construct():
    n, m, p = 3, 4, 8
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t = i * (m * p) + j * p + k
                U[i * m + j, t] = 1.0
                V[j * p + k, t] = 1.0
                W[k * n + i, t] = 1.0
    return {'U': U, 'V': V, 'W': W}
