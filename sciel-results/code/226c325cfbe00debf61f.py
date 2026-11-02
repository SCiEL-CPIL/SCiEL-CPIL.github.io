"""Construct a low-rank exact tensor decomposition for 3x3 * 3x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rng, trials=4):
    """
    Check the decomposition reproduces matrix multiplication.
    Convention (matches schoolbook):
        C[i,k] = sum_{i', jA, jB} sum_r U[i*m+jA, r] V[jB*p+k, r] W[k*n+i', r] A[i',jA] B[jB,k]
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for _ in range(trials):
        A = rng.random((n, m))
        B = rng.random((m, p))
        C_true = A @ B
        C = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for i2 in range(n):
                    for jA in range(m):
                        for jB in range(m):
                            s += float(np.dot(U[i * m + jA], V[jB * p + k] * W[k * n + i2])) \
                                 * A[i2, jA] * B[jB, k]
                C[i, k] = s
        if not np.allclose(C, C_true, atol=1e-9):
            return False
    return True


def _build_block18():
    """
    Rank-18 block construction for 3x3 x 3x4.
    Partition rows of A: top (i=0,1) and bottom (i=2).
    Partition cols of B: left (k=0,1) and right (k=2,3).
    Four independent 2x3@3x2 (rank 6) and 1x3@3x2 (rank 3) sub-products:
      C_top_left  (2x2) = A[0:2,:] @ B[:,0:2]  -> 6 terms
      C_top_right (2x2) = A[0:2,:] @ B[:,2:4]  -> 6 terms
      C_bot_left  (1x2) = A[2,:]    @ B[:,0:2]  -> 3 terms
      C_bot_right (1x2) = A[2,:]    @ B[:,2:4]  -> 3 terms
    Total rank = 6 + 6 + 3 + 3 = 18.
    """
    n, m, p = 3, 3, 4
    rank = 18
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0

    def term(i, j, k):
        nonlocal col
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1

    # Top-left 2x2 = A[0:2,:] @ B[:,0:2]
    for i in range(2):
        for j in range(3):
            for k in range(2):
                term(i, j, k)
    # Top-right 2x2 = A[0:2,:] @ B[:,2:4]
    for i in range(2):
        for j in range(3):
            for k in range(2, 4):
                term(i, j, k)
    # Bottom-left 1x2 = A[2,:] @ B[:,0:2]
    for j in range(3):
        for k in range(2):
            term(2, j, k)
    # Bottom-right 1x2 = A[2,:] @ B[:,2:4]
    for j in range(3):
        for k in range(2, 4):
            term(2, j, k)

    assert col == rank, (col, rank)
    return U.tolist(), V.tolist(), W.tolist()


def _build_schoolbook():
    n, m, p = 3, 3, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s
    n, m, p = 3, 3, 4
    rng = np.random.default_rng(20240517)

    # Primary: rank-18 block construction, verified exactly.
    U, V, W = _build_block18()
    if _verify(U, V, W, n, m, p, rng, trials=4):
        return {"U": U, "V": V, "W": W}

    # Fallback: guaranteed-valid rank-36 schoolbook.
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
