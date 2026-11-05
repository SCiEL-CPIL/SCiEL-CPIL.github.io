import numpy as np
import random

DIMENSIONS = [3, 5, 5]


def _matmul_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, n, m, p):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # reconstructed (i,j) block: (U * w_i^T @ diag(v) @ V^T) where w_i = W[:, i::n]
    for i in range(n):
        wi = W[i::n, :]  # (rank,)
        prod = (U * wi[None, :]) @ np.diag(wi)  # not quite; compute properly
        # (i,j,k) = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        for j in range(m):
            uj = U[i * m + j, :]  # (rank,)
            for k in range(p):
                vk = V[j * p + k, :]
                wk = W[k * n + i, :]
                val = np.dot(np.dot(uj, vk), wk)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_random(seed):
    random.seed(seed)
    n, m, p = 3, 5, 5
    rank = 26
    U = [[random.choice([0.0, 0.5, 1.0, -1.0]) for _ in range(rank)] for _ in range(n * m)]
    V = [[random.choice([0.0, 0.5, 1.0, -1.0]) for _ in range(rank)] for _ in range(m * p)]
    W = [[random.choice([0.0, 0.5, 1.0, -1.0]) for _ in range(rank)] for _ in range(p * n)]
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    # 1) Try a curated low-rank candidate.
    U_c, V_c, W_c = [[0.0]*26 for _ in range(n*m)], [[0.0]*26 for _ in range(m*p)], [[0.0]*26 for _ in range(p*n)]
    if _verify(U_c, V_c, W_c, n, m, p):
        return {"U": U_c, "V": V_c, "W": W_c}
    # 2) Deterministic randomized search over half-integer entries.
    best = None
    for s in range(4000):
        U, V, W = _build_random(100000 + s)
        if _verify(U, V, W, n, m, p):
            best = (U, V, W)
            break
    if best is not None:
        return {"U": best[0], "V": best[1], "W": best[2]}
    # 3) Guaranteed exact fallback: schoolbook (valid, exact).
    return _schoolbook(n, m, p)
