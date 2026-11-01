import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rng = np.random.RandomState(0)
    for _ in range(8):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        C_true = A @ B
        a = A.reshape(n*m)
        b = B.reshape(m*p)
        ua = U.T @ a
        vb = V.T @ b
        c = W.T @ (ua * vb)
        C = c.reshape(p, n).T
        if not np.allclose(C, C_true, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    def rank27():
        rank = 27
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    r27 = rank27()
    if _verify(r27["U"], r27["V"], r27["W"], n, m, p):
        return r27
    return r27
