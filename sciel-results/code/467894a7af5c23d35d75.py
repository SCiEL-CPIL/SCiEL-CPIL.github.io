import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    T = np.einsum('uc,vc,wc->uvw', U, V, W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(T[u, v, w] - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    U_out = [[(int(x * 2) / 2) for x in row] for row in best_U]
    V_out = [[(int(x * 2) / 2) for x in row] for row in best_V]
    W_out = [[(int(x * 2) / 2) for x in row] for row in best_W]

    return {"U": U_out, "V": V_out, "W": W_out}
