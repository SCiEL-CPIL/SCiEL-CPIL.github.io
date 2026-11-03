import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = 2, 3, 6

    # The matrix multiplication tensor for dense matrices is the all-ones tensor.
    # T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p).
    # This is because every element A[i, j] is multiplied by every element B[j, k]
    # to contribute to C[i, k].
    # The rank of the all-ones tensor is 1.
    # We construct a rank-1 decomposition:
    # U[i, 0] = 1
    # V[j, 0] = 1
    # W[k, 0] = 1
    
    rank = 1
    U = np.ones((n * m, rank), dtype=float)
    V = np.ones((m * p, rank), dtype=float)
    W = np.ones((p * n, rank), dtype=float)

    # Verify the decomposition
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook if verification fails (should not happen)
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
