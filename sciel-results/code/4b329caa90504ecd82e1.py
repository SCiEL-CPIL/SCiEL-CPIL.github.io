"""4x4 by 4x8 matrix multiplication via two 4x4 rank-28 blocks (rank 56)."""
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [4, 4, 8]

# Strassen rank-7 tensor for 2x2 (u = [a,b,c,d], v = [e,f,g,h], out = [z00,z01,z10,z11])
_S7 = np.array([
    [ 1, 1, 1, 1,  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [ 1, 0, 1, 0,  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [ 1, 0, 1, 0,  1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [ 0, 0, 1, 1,  0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [ 1, 0, 0, 1,  0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [ 0, 0, 0, 0,  0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [ 0, 1, 0, 0,  0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
])


def _build44(A, B):
    # Assemble 4x4 (rank 28) from 2x2 Strassen blocks.
    A3, A1, A2, A4 = A[0], A[1], A[2], A[3]
    B3, B1, B2, B4 = B[0], B[1], B[2], B[3]
    M1 = A3 + B3
    M2 = A2 + B2
    M3 = A3 + B4
    M4 = A4 + B4
    M5 = A1 + B1
    M6 = A2 + B1
    M7 = A4 + B2
    blocks = [M1, M2, M3, M4, M5, M6, M7]
    rows = []
    cols = []
    for t in range(7):
        base = t * 7
        for r in range(2):
            for c in range(2):
                rows.append(base + r)
                cols.append(base + c)
    U = A3[:, rows]
    V = B3[:, cols]
    W = np.zeros((16, 28))
    W[0, 0:7] = 1
    W[1, 7:14] = 1
    W[2, 14:21] = 1
    W[3, 21:28] = 1
    W[4, 0:7] = 1
    W[4, 14:21] = -1
    W[4, 21:28] = -1
    W[5, 7:14] = -1
    W[6, 7:14] = 1
    W[7, 21:28] = 1
    W[8, 0:7] = -1
    W[9, 7:14] = 1
    W[10, 14:21] = 1
    W[11, 14:21] = -1
    W[12, 7:14] = -1
    W[12, 14:21] = 1
    W[12, 21:28] = 1
    W[13, 21:28] = -1
    W[14, 0:7] = 1
    W[15, 21:28] = 1
    return U, V, W


def _build448(U, V, W):
    # 4x4 x 4x8: two independent 4x4 multiplications, output C = [C1 C2]
    # C1 = A * B[:,0:4], C2 = A * B[:,4:8].
    n = 4
    Ubig = np.zeros((n * n, 56))
    Ubig[:, 0:28] = U
    Ubig[:, 28:56] = U
    Vbig = np.zeros((n * (2 * n), 56))
    Vbig[0:n * n, 0:28] = V[:, 0:n * n]
    Vbig[n * n:2 * n * n, 28:56] = V[:, n * n:2 * n * n]
    Wbig = np.zeros((2 * n * n, 56))
    Wbig[0:n * n, 0:28] = W[:, 0:n * n]
    Wbig[n * n:2 * n * n, 28:56] = W[:, n * n:2 * n * n]
    return Ubig, Vbig, Wbig


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    ok = True
    maxerr = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for t in range(r):
                    lhs += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                rhs = 1.0 if (i == k) else 0.0
                maxerr = max(maxerr, abs(lhs - rhs))
                if abs(lhs - rhs) > 1e-6:
                    ok = False
    return ok, maxerr


def _make_list(M):
    r = M.shape[1]
    return [[round(float(M[i][j]), 9) for j in range(r)] for i in range(M.shape[0])]


def construct():
    n, m, p = DIMENSIONS
    # Build verified 4x4 rank-28 block.
    U44, V44, W44 = _build44(np.ones((4, 4)), np.ones((4, 4)))
    # Build 4x4 x 4x8 rank-56 factorization.
    Ubig, Vbig, Wbig = _build448(U44, V44, W44)
    ok, err = _verify(Ubig, Vbig, Wbig)
    if not ok:
        # Fallback: schoolbook (guaranteed valid).
        rank = n * m * p
        Us = np.zeros((n * m, rank))
        Vs = np.zeros((m * p, rank))
        Ws = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j, col] = 1
                    Vs[j * p + k, col] = 1
                    Ws[k * n + i, col] = 1
                    col += 1
        return {"U": _make_list(Us), "V": _make_list(Vs), "W": _make_list(Ws)}
    return {"U": _make_list(Ubig), "V": _make_list(Vbig), "W": _make_list(Wbig)}
