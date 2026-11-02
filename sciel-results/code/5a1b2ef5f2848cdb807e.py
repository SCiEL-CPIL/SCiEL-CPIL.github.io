import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = []
    V = []
    W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    return {"U": U, "V": V, "W": W}
