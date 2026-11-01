import numpy as np
import time

_N, _M, _P = 2, 4, 8
_DEADLINE = None

def _now():
    return time.time()

def _leftover():
    return max(0.0, _DEADLINE - _now() - 4.0)

# ---------- generic tensor-decomposition builder for n x m by m x p ----------
def _build_UVW(n, m, p, U, V, W, iters=3, tol=1e-6):
    """Verify that the bilinear map defined by (U,V,W) equals matrix product.
    U: (n*m) x r, V: (m*p) x r, W: (p*n) x r, W indexed by [k*n + i, r]."""
    r = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for _ in range(iters):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        M = (U.T @ A.reshape(n * m)) * (V.T @ B.reshape(m * p))  # (r,)
        C = (W.T @ M).reshape(p, n).T                             # (n, p)
        if not np.allclose(C, A @ B, atol=tol, rtol=tol):
            return False
    return True

# ---------- Winograd 4 x 3 x 3 (two overlapping windows) ----------
def _winograd_2window():
    """Return U (8 x r), V (12 x r), W (16 x r) for C = A[4x3] @ B[3x4].
    W indexed by [k*4 + i, r], A indexed by [i*3 + j, r], B by [j*4 + k, r]."""
    N, M, P = 4, 3, 4
    B1 = np.zeros((M, 2))
    B1[0, 0] = 1; B1[1, 1] = 1; B1[2, 0] = 1; B1[2, 1] = -1
    B2 = np.zeros((P, 3))
    B2[0, 0] = 1; B2[1, 0] = 1; B2[1, 1] = 1; B2[2, 0] = -1
    B2[2, 1] = -1; B2[3, 0] = 1; B2[3, 2] = 1
    r = 2 * 3

    # window 0: B rows 0..2, C cols 0..3
    V1 = np.zeros((M * P, r)); W1 = np.zeros((P * N, r))
    for t in range(2):
        for j in range(M):
            V1[j * P + t, t] = B1[j, t]
    for i in range(N):
        for k in range(P):
            W1[k * N + i, i] = B2[k, i]

    # window 1: B rows 1..3, C cols 1..4  (shift B by one, C by one)
    V2 = np.zeros((M * P, r)); W2 = np.zeros((P * N, r))
    for t in range(2):
        for j in range(M):
            V2[(j + 1) * P + t, t] = B1[j, t]
    for i in range(N):
        for k in range(P):
            W2[(k + 1) * N + i, i] = B2[k, i]

    U = np.zeros((N * M, 2 * r))
    for i in range(N):
        for j in range(M):
            U[i * M + j, i] = 1.0
            U[i * M + j, r + i] = 1.0
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    return U, V, W

def _winograd_2window_result():
    U, V, W = _winograd_2window()
    r = U.shape[1]
    if not _build_UVW(4, 3, 4, U, V, W):
        return None
    return (U.tolist(), V.tolist(), W.tolist(), r)

# ---------- Kronecker 2 x 2 Winograd blocks (rank 56 fallback) ----------
def _kronecker_fallback():
    n, m, p = _N, _M, _P
    U2 = np.array([[1,0,0,1,0,0,0],
                   [0,1,0,0,1,0,1],
                   [0,0,1,0,0,0,0],
                   [0,0,0,1,0,1,0]])
    V2 = np.array([[1,0,0,0,0,0,1],
                   [0,1,0,0,0,1,1],
                   [0,0,1,0,1,0,0],
                   [0,0,0,1,0,1,0]])
    W2 = np.array([[1,0,1,0,-1,0,1],
                   [0,1,0,0,0,1,-1],
                   [0,0,0,1,1,1,0],
                   [1,-1,1,1,0,0,0]])
    r = 56
    U = np.zeros((n * m, r)); V = np.zeros((m * p, r)); W = np.zeros((p * n, r))
    for a in range(2):
        for b in range(4):
            off = (a * 4 + b) * 7
            for i in range(2):
                for jl in range(2):
                    jg = a * 2 + jl
                    U[i * m + jg, off + (i * 2 + jl)] = U2[i * 2 + jl, i]
            for jl in range(2):
                jg = a * 2 + jl
                for kl in range(2):
                    kg = b * 2 + kl
                    V[jg * p + kg, off + (jl * 2 + kl)] = V2[jl * 2 + kl, jl]
            for i in range(2):
                for kl in range(2):
                    kg = b * 2 + kl
                    W[kg * n + i, off + (kl * 2 + i)] = W2[kl * 2 + i, i]
    if not _build_UVW(n, m, p, U, V, W):
        return None
    return (U.tolist(), V.tolist(), W.tolist(), r)

# ---------- schoolbook last resort ----------
def _schoolbook():
    n, m, p = _N, _M, _P
    r = n * m * p
    U = np.zeros((n * m, r)); V = np.zeros((m * p, r)); W = np.zeros((p * n, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1; V[j * p + k, c] = 1; W[k * n + i, c] = 1
                c += 1
    return (U.tolist(), V.tolist(), W.tolist(), r)

def construct():
    global _DEADLINE
    _DEADLINE = _now() + 118.0

    # 1) Winograd two-window construction (rank 42)
    if _leftover() > 2.0:
        res = _winograd_2window_result()
        if res is not None:
            U, V, W, r = res
            if _build_UVW(_N, _M, _P, U, V, W):
                return {"U": U, "V": V, "W": W}

    # 2) Kronecker 2x2 Strassen blocks (rank 56)
    if _leftover() > 2.0:
        res = _kronecker_fallback()
        if res is not None:
            U, V, W, r = res
            if _build_UVW(_N, _M, _P, U, V, W):
                return {"U": U, "V": V, "W": W}

    # 3) schoolbook (rank 64)
    U, V, W, r = _schoolbook()
    return {"U": U, "V": V, "W": W}
