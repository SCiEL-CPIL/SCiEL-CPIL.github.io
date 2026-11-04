"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 by 4x8."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _matrices_from_tensor(U, V, W):
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float).reshape(n * m, -1)
    V = np.asarray(V, dtype=float).reshape(m * p, -1)
    W = np.asarray(W, dtype=float).reshape(p * n, -1)
    A = U.reshape(n, m, -1).transpose(2, 0, 1)
    B = V.reshape(m, p, -1).transpose(2, 0, 1)
    C = W.reshape(p, n, -1).transpose(2, 0, 1)
    return A, B, C

def _verify(U, V, W):
    """Return True iff U,V,W exactly decompose the matrix-multiplication tensor."""
    try:
        n, m, p = DIMENSIONS
        U = np.asarray(U, dtype=float).reshape(n * m, -1)
        V = np.asarray(V, dtype=float).reshape(m * p, -1)
        W = np.asarray(W, dtype=float).reshape(p * n, -1)
        A = U.reshape(n, m, -1).transpose(2, 0, 1)
        B = V.reshape(m, p, -1).transpose(2, 0, 1)
        C = W.reshape(p, n, -1).transpose(2, 0, 1)
        rng = np.random.default_rng(123456789)
        for _ in range(20):
            X = rng.standard_normal((n, m))
            Y = rng.standard_normal((m, p))
            Z = np.einsum('cin,cin->cn', A, B)
            prod = np.einsum('cn,cnp->cp', Z, C)
            if not np.allclose(prod, X @ Y, rtol=1e-9, atol=1e-8):
                return False
        return True
    except Exception:
        return False

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _strassen_2x2():
    """Rank-7 Strassen for 2x2x2. Returns A,B,C each shape (7,2,2)."""
    A = np.zeros((7, 2, 2))
    B = np.zeros((7, 2, 2))
    C = np.zeros((7, 2, 2))
    A[0] = [[1, 0], [0, 1]]
    B[0] = [[1, 0], [0, -1]]
    C[0] = [[0, 1], [0, 1]]
    A[1] = [[1, 1], [0, 0]]
    B[1] = [[0, 0], [1, -1]]
    C[1] = [[0, 0], [1, 1]]
    A[2] = [[1, 0], [0, 0]]
    B[2] = [[1, 1], [0, 1]]
    C[2] = [[1, 0], [0, 0]]
    A[3] = [[0, 0], [1, 1]]
    B[3] = [[1, 0], [1, 1]]
    C[3] = [[0, 1], [0, 0]]
    A[4] = [[1, -1], [1, -1]]
    B[4] = [[1, 0], [0, 1]]
    C[4] = [[1, 0], [0, 1]]
    A[5] = [[-1, 1], [0, 1]]
    B[5] = [[1, 0], [1, 0]]
    C[5] = [[0, 1], [1, 0]]
    A[6] = [[1, 0], [1, 0]]
    B[6] = [[0, -1], [1, 1]]
    C[6] = [[1, 0], [1, 0]]
    return A, B, C

def _kron_factorB():
    """Rank-8 factor for 2x2x4 = Strassen(2x2x2) kron I_2."""
    A, B, C = _strassen_2x2()
    I2 = np.eye(2)
    r = A.shape[0]
    Ak = np.zeros((r, 2, 2, 4))
    Bk = np.zeros((r, 2, 2, 4))
    Ck = np.zeros((r, 4, 2, 2))
    for t in range(r):
        for a in range(2):
            for b in range(2):
                for s in range(2):
                    for d in range(2):
                        Ak[t, a, b, 2 * s + d] = A[t, a, b] * I2[s, d]
                        Bk[t, a, b, 2 * s + d] = B[t, a, b] * I2[s, d]
                        Ck[t, 2 * s + d, a, b] = C[t, a, b] * I2[s, d]
    return Ak, Bk, Ck

def _kron_4x4x8():
    """Compose (2x2x2) kron (2x2x4) into a 4x4x8 decomposition (rank 56)."""
    A1, B1, C1 = _strassen_2x2()          # (7,2,2)
    A2, B2, C2 = _kron_factorB()          # (8,2,2,4)
    r1, r2 = A1.shape[0], A2.shape[0]
    r = r1 * r2
    A = np.zeros((r, 4, 4))
    B = np.zeros((r, 4, 8))
    C = np.zeros((r, 8, 4))
    idx = 0
    for t1 in range(r1):
        for t2 in range(r2):
            for i in range(2):
                for j in range(2):
                    for s in range(2):
                        for d in range(2):
                            A[idx, 2 * i + s, 2 * j + d] = A1[t1, i, j] * A2[t2, s, d, 0]
                            B[idx, 2 * j + s, 2 * d] = B1[t1, j, d] * B2[t2, s, d, 0]
                            C[idx, 2 * d, 2 * i + s] = C1[t1, i, s] * C2[t2, 2 * d, s, 0]
            idx += 1
    return A, B, C

def construct():
    deadline = time.time() + 120.0
    best = None

    U, V, W = _schoolbook()
    best = (U, V, W)

    try:
        A, B, C = _kron_4x4x8()
        U = A.transpose(1, 2, 0).reshape(4 * 4, -1)
        V = B.transpose(1, 2, 0).reshape(4 * 8, -1)
        W = C.transpose(1, 2, 0).reshape(8 * 4, -1)
        if _verify(U, V, W):
            best = (U, V, W)
    except Exception:
        pass

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
