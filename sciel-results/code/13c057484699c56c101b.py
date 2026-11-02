import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((n*p, rank))
    for r in range(rank):
        i = r // (m*p)
        j = (r // p) % m
        k = r % p
        U[i*m+j, r] = 1.0
        V[j*p+k, r] = 1.0
        W[i*p+k, r] = 1.0
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
