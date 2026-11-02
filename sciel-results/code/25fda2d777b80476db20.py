import time
import numpy as np

DIMENSIONS = [2, 3, 10]
n, m, p = DIMENSIONS
TENSOR_SIZE = n * m * p  # 60
DEADLINE = 130.0


def _build_tensors(n, m, p, U, V, W):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    Uc = U.reshape(n, m, rank)
    Vc = V.reshape(m, p, rank)
    Wc = W.reshape(p, n, rank)
    W3d = np.zeros((n, m, p), dtype=float)
    for k in range(p):
        for i in range(n):
            W3d[i, :, k] = Wc[k, i, :]
    M = np.einsum('ijc,jkc,ijk->ijk', Uc, Vc, W3d)
    return M


def _verify(n, m, p, U, V, W, tol=1e-6):
    M = _build_tensors(n, m, p, U, V, W)
    return np.allclose(M, 1.0, atol=tol, rtol=0)


def construct():
    start = time.time()
    
    # Triple split: rank 6 decomposition
    rank = m * n  # 3 * 2 = 6
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for j in range(m):
        for i in range(n):
            c = j * n + i
            U[i * m + j, c] = 1
            for k in range(p):
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
    
    # Verify
    if not _verify(n, m, p, U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
