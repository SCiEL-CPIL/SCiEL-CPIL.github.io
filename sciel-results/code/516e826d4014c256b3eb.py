import numpy as np
import time

def construct():
    n, m, p = 2, 2, 14
    start_time = time.time()
    deadline = start_time + 120
    size_a = n * m
    size_b = m * p
    size_c = p * n

    T_true = np.zeros((size_a, size_b, size_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, atol=1e-9):
        Ua = np.asarray(U, dtype=np.float64)
        Va = np.asarray(V, dtype=np.float64)
        Wa = np.asarray(W, dtype=np.float64)
        T = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
        return np.allclose(T, T_true, atol=atol)

    def strassen49():
        rank = 49
        U = np.zeros((size_a, rank), dtype=np.int64)
        V = np.zeros((size_b, rank), dtype=np.int64)
        W = np.zeros((size_c, rank), dtype=np.int64)

        for l in range(7):
            base = 7 * l
            k0 = 2 * l
            k1 = 2 * l + 1

            c0 = base + 0
            U[0, c0] += 1; U[3, c0] += 1
            V[k0, c0] += 1; V[p + k1, c0] += 1
            W[k0 * n, c0] += 1; W[k1 * n + 1, c0] += 1

            c1 = base + 1
            U[2, c1] += 1; U[3, c1] += 1
            V[k0, c1] += 1
            W[k0 * n + 1, c1] += 1; W[k1 * n + 1, c1] -= 1

            c2 = base + 2
            U[0, c2] += 1
            V[k1, c2] += 1; V[p + k1, c2] -= 1
            W[k1 * n, c2] += 1; W[k1 * n + 1, c2] += 1

            c3 = base + 3
            U[3, c3] += 1
            V[p + k0, c3] += 1; V[k0, c3] -= 1
            W[k0 * n, c3] += 1; W[k0 * n + 1, c3] += 1

            c4 = base + 4
            U[0, c4] += 1; U[1, c4] += 1
            V[p + k1, c4] += 1
            W[k0 * n, c4] -= 1; W[k1 * n, c4] += 1

            c5 = base + 5
            U[2, c5] += 1; U[0, c5] -= 1
            V[k0, c5] += 1; V[k1, c5] += 1
            W[k1 * n + 1, c5] += 1

            c6 = base + 6
            U[1, c6] += 1; U[3, c6] -= 1
            V[p + k0, c6] += 1; V[p + k1, c6] += 1
            W[k0 * n, c6] += 1

        return U, V, W

    U, V, W = strassen49()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    rank = n * m * p
    U = np.zeros((size_a, rank), dtype=np.int64)
    V = np.zeros((size_b, rank), dtype=np.int64)
    W = np.zeros((size_c, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
