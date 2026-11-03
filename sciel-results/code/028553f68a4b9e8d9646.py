import numpy as np
import time

DIMENSIONS = [3, 5, 6]


def _strassen_2x2x2():
    """Return U,V,W (each (4,7)) for the 2x2x2 multiplication tensor (Strassen, rank 7).
    U[i*2+j, c] for A[i,j]; V[j*2+k, c] for B[j,k]; W[k*2+i, c] for C[i,k]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # A factors
    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    U[1, 6] = 1; U[3, 6] = -1
    # B factors
    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[1, 2] = 1; V[3, 2] = -1
    V[2, 3] = 1; V[0, 3] = -1
    V[3, 4] = 1
    V[0, 5] = 1; V[1, 5] = 1
    V[2, 6] = 1; V[3, 6] = 1
    # C factors (W[k*2+i, c])
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[2, 2] = 1; W[2, 4] = 1
    W[1, 1] = 1; W[1, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    return U, V, W


def _build_base(n, m, p):
    """Build the base decomposition: 2x2x2 Strassen blocks + schoolbook strips.
    Returns U,V,W as (n*m, R), (m*p, R), (p*n, R) float arrays."""
    R = 84
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    c0 = _strassen_2x2x2()
    cu, cv, cw = c0[0], c0[1], c0[2]

    # 2x2x2 Strassen blocks: i0 in {0,1}, k0 in {0,2,4}
    col = 0
    for i0 in (0, 1):
        for k0 in (0, 2, 4):
            for r in range(7):
                for di in (0, 1):
                    U[(i0 + di) * m, col] += cu[di, r]
                for dk in (0, 1):
                    W[(k0 + dk) * n + i0, col] += cw[dk, r]
            for j in range(5):
                for r in range(7):
                    V[j * p + k0, col] += cv[j, r]
                    V[j * p + k0 + 1, col] += cv[j + 2, r]
            col += 7

    # Schoolbook strips: row i=2 (j,k all) and column k=5 (i in {0,1}, j all)
    for j in range(5):
        for k in range(6):
            U[2 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 2, col] = 1.0
            col += 1
    for i in (0, 1):
        for j in range(5):
            U[i * m + j, col] = 1.0
            V[j * p + 5, col] = 1.0
            W[5 * n + i, col] = 1.0
            col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                if abs(s - 1.0) > 1e-8:
                    return False
    return True


def _schoolbook(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _refine(U, V, W, n, m, p):
    """Try to drop rank terms that are linear combinations of the others.
    Returns (U', V', W') with minimal rank found."""
    R = U.shape[1]
    best = (U, V, W)
    R_best = R
    # Order: try removing strip terms (col >= 42) first, then blocks
    order = list(range(42, R)) + list(range(0, 42))
    for c in order:
        if R_best <= 2:
            break
        U2 = np.delete(U, c, axis=1)
        V2 = np.delete(V, c, axis=1)
        W2 = np.delete(W, c, axis=1)
        if _verify(U2, V2, W2, n, m, p):
            U, V, W = U2, V2, W2
            R_best = R_best - 1
            best = (U, V, W)
    return best


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s wall-clock

    U, V, W = _build_base(n, m, p)
    if _verify(U, V, W, n, m, p):
        if time.time() < deadline:
            U, V, W = _refine(U, V, W, n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
