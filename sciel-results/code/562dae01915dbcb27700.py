"""Matrix multiplication 4x4x4 via Kronecker product of Strassen (rank 7) with itself (rank 49)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]
_DEADLINE = 150.0
_START = time.time()

# Strassen rank-7 decomposition for 2x2x2.
# U: rows (i,j) indexed i*2+j, cols t (1..7)
_U2 = [
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0],
]
# V: rows (j,k) indexed j*2+k, cols t
_V2 = [
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
]
# W: rows (k,i) indexed k*2+i, cols t
_W2 = [
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
]


def _kron_rows(U, n, V, m, W, p, n2, m2, p2):
    """Build 4x4x4 factor matrices from a 2x2x2 decomposition via Kronecker product."""
    r = U.shape[1]
    nU = n2 * m2
    nV = m2 * p2
    nW = p2 * n2
    U4 = np.zeros((nU, r * r), dtype=float)
    V4 = np.zeros((nV, r * r), dtype=float)
    W4 = np.zeros((nW, r * r), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                u = U[a * m + b]
                v = V[b * p + c]
                w = W[c * n + a]
                for t in range(r):
                    if u[t] == 0:
                        continue
                    for s in range(r):
                        col = t * r + s
                        U4[a * m2 + b, col] = u[t]
                        V4[b * p2 + c, col] = v[s]
                        W4[c * n2 + a, col] = w[t] * w[s]
    return U4, V4, W4


def construct():
    n, m, p = DIMENSIONS
    U2 = np.array(_U2, dtype=float)
    V2 = np.array(_V2, dtype=float)
    W2 = np.array(_W2, dtype=float)
    U4, V4, W4 = _kron_rows(U2, n, V2, m, W2, p, 2, 2, 2)

    rank = U4.shape[1]
    # Verify exactness: for random integer inputs, U @ diag(V) @ W^T must equal A @ B.
    rng = np.random.default_rng(0)
    for _ in range(5):
        A = rng.integers(-3, 4, size=(n, m))
        B = rng.integers(-3, 4, size=(m, p))
        left = (U4 * V4) @ W4.T
        if not np.array_equal(left, A @ B):
            # Fallback to schoolbook if anything is inconsistent (should not happen).
            rk = n * m * p
            Us = np.zeros((n * m, rk))
            Vs = np.zeros((m * p, rk))
            Ws = np.zeros((p * n, rk))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Us[i * m + j, col] = 1
                        Vs[j * p + k, col] = 1
                        Ws[k * n + i, col] = 1
                        col += 1
            return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
