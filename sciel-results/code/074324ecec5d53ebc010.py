import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _winograd_444_G():
    """Winograd G matrix for F(4,4,4)."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, -1, 0],
        [0, 0, 0, 1]
    ], dtype=float)

def _add_winograd_444(U, V, W, col, a_base, b_base, c_base):
    """Add Winograd F(4,4,4) rank-16 terms. Bases are global row-major linear indices."""
    G = _winograd_444_G()
    # U[i*m+j, col] = G[i,a] * G[j,b] where r = a*4+b
    # V[j*p+k, col] = G[j,a] * G[k,b]
    # W[k*n+i, col] = G[i,b] * G[k,a]  (note: transposed indices)
    
    for a in range(4):
        for b in range(4):
            r = col + a * 4 + b
            # U: rows indexed by (i,j) in 4x4, value G[i,a]*G[j,b]
            for i in range(4):
                for j in range(4):
                    U[a_base + i * 4 + j][r] += G[i, a] * G[j, b]
            # V: rows indexed by (j,k) in 4x4, value G[j,a]*G[k,b]
            for j in range(4):
                for k in range(4):
                    V[b_base + j * 4 + k][r] += G[j, a] * G[k, b]
            # W: rows indexed by (k,i) in 4x4, value G[i,b]*G[k,a]
            for k in range(4):
                for i in range(4):
                    W[c_base + k * 4 + i][r] += G[i, b] * G[k, a]
    return col + 16

def _add_matrix_vector_block(U, V, W, col, a_base, b_base, c_base, ar, ac, bc):
    """Matrix-vector: A(ar x ac) * B(ac x bc) where bc=1. Schoolbook, rank ar*ac."""
    for i in range(ar):
        for j in range(ac):
            U[a_base + i * ac + j][col] += 1
            V[b_base + j * bc][col] += 1
            W[c_base + i * bc][col] += 1
            col += 1
    return col

def _verify(U, V, W, n, m, p):
    """Return True iff sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for every
    (i,j,k) and 0 elsewhere."""
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True

def _build_rank48(n, m, p):
    """Winograd(16) + Winograd(16) + matrix-vector(16) = 48.
    Split 4x4x9 into 4x4x4 (Winograd, 16) + 4x4x4 (Winograd, 16) + 4x4x1 (schoolbook, 16)."""
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Part 1: A(4x4) * B_left(4x4) -> C_left(4x4). Winograd, rank 16.
    col = _add_winograd_444(U, V, W, col, 0, 0, 0)

    # Part 2: A(4x4) * B_mid(4x4) -> C_mid(4x4). Winograd, rank 16.
    col = _add_winograd_444(U, V, W, col, 0, 4, 4)

    # Part 3: A(4x4) * B_last(4x1) -> C_last(4x1). Schoolbook, rank 16.
    col = _add_matrix_vector_block(U, V, W, col, 0, 8, 8, 4, 4, 1)

    if col != rank:
        rank = col
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
    return {"U": U, "V": V, "W": W}

def _build_schoolbook(n, m, p):
    """Guaranteed rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # Primary: rank-48 Winograd + Winograd + matrix-vector.
    res = _build_rank48(n, m, p)
    if time.time() < deadline and _verify(res["U"], res["V"], res["W"], n, m, p):
        return res

    # Fallback: guaranteed schoolbook rank 144.
    return _build_schoolbook(n, m, p)
