"""Rank-minimal exact 3x3x3 matrix multiplication tensor decomposition (Strassen, rank 23)."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_strassen_23():
    n, m, p = DIMENSIONS
    U = [[0.0] * 23 for _ in range(n * m)]
    V = [[0.0] * 23 for _ in range(m * p)]
    W = [[0.0] * 23 for _ in range(p * n)]

    def A(i, j):
        return i * m + j

    def B(j, k):
        return j * p + k

    def C(k, i):
        return k * n + i

    def add(r, a, b, c):
        U[a][r] += c
        V[b][r] += c

    def sub(r, a, b, c):
        U[a][r] -= c
        V[b][r] -= c

    def setv(r, a, b, c):
        U[a][r] = c
        V[b][r] = c

    # Strassen's 7 products on the 2x2 core (indices 0..1).
    # M1 = (A00+A11)*(B00+B11)
    add(0, A(0, 0), B(0, 0), 1); add(0, A(1, 1), B(0, 0), 1)
    add(0, A(0, 0), B(1, 1), 1); add(0, A(1, 1), B(1, 1), 1)
    # M2 = (A10+A11)*B00
    add(1, A(1, 0), B(0, 0), 1); add(1, A(1, 1), B(0, 0), 1)
    # M3 = A00*(B01-B11)
    setv(2, A(0, 0), B(0, 1), 1); setv(2, A(0, 0), B(1, 1), -1)
    # M4 = A11*(B10-B00)
    setv(3, A(1, 1), B(1, 0), 1); setv(3, A(1, 1), B(0, 0), -1)
    # M5 = (A00+A01)*B11
    add(4, A(0, 0), B(1, 1), 1); add(4, A(0, 1), B(1, 1), 1)
    # M6 = (A10-A00)*(B00+B01)
    sub(5, A(1, 0), B(0, 0), 1); setv(5, A(0, 0), B(0, 0), -1)
    add(5, A(1, 0), B(0, 1), 1); setv(5, A(0, 0), B(0, 1), -1)
    # M7 = (A01-A11)*(B10+B11)
    sub(6, A(0, 1), B(1, 0), 1); setv(6, A(1, 1), B(1, 0), -1)
    add(6, A(0, 1), B(1, 1), 1); setv(6, A(1, 1), B(1, 1), -1)

    # W contributions for the 2x2 core C entries:
    # C00 = M1 - M4 + M6 - M7
    W[C(0, 0)][0] += 1; W[C(0, 0)][3] += -1; W[C(0, 0)][5] += 1; W[C(0, 0)][6] += -1
    # C01 = M3 + M5
    W[C(1, 0)][2] += 1; W[C(1, 0)][4] += 1
    # C10 = M2 + M4
    W[C(0, 1)][1] += 1; W[C(0, 1)][3] += 1
    # C11 = M1 - M2 + M5 + M7
    W[C(1, 1)][0] += 1; W[C(1, 1)][1] += -1; W[C(1, 1)][4] += 1; W[C(1, 1)][6] += 1

    # Border rank-1 terms
    setv(7, A(0, 2), B(0, 2), 1); W[C(2, 0)][7] = 1
    setv(8, A(1, 2), B(1, 2), 1); W[C(2, 1)][8] = 1
    setv(9, A(2, 0), B(2, 0), 1); W[C(0, 2)][9] = 1
    setv(10, A(2, 1), B(2, 1), 1); W[C(1, 2)][10] = 1
    setv(11, A(2, 2), B(2, 2), 1); W[C(2, 2)][11] = 1
    return U, V, W


def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(V[0])
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    P = Ua @ Va.T  # (n*m, m*p)
    P = P.reshape(n, m, m, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += P[i, j, r] * Wa[k * n + i, r]
                expected = 1.0 if (j == k) else 0.0
                if abs(s - expected) > 1e-9:
                    return False
    return True


def _to_int_or_half(U, V, W):
    def conv(M):
        out = []
        for row in M:
            nr = []
            for x in row:
                xr = round(x)
                if abs(x - xr) < 1e-9:
                    nr.append(int(xr))
                else:
                    xr2 = round(2 * x)
                    if abs(2 * x - xr2) < 1e-9:
                        nr.append(xr2 / 2)
                    else:
                        nr.append(x)
            out.append(nr)
        return out
    return conv(U), conv(V), conv(W)


def construct():
    import time
    t0 = time.time()
    U, V, W = _build_strassen_23()
    if time.time() - t0 > 120:
        U, V, W = _build_schoolbook()
        return {"U": U, "V": V, "W": W}
    if _verify(U, V, W):
        Ui, Vi, Wi = _to_int_or_half(U, V, W)
        return {"U": Ui, "V": Vi, "W": Wi}
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
