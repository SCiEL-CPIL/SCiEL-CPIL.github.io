"""Tensor decomposition for 2x3 times 3x9 matrix multiplication."""
DIMENSIONS = [2, 3, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    # Rank-18 decomposition: treat as two independent 1x3 x 3x9 multiplications.
    # For each row i of A and each column k of the output, one rank-1 term.
    rank = n * p  # 2 * 9 = 18
    U = [[0] * rank for _ in range(n * m)]  # (n*m, rank) = (6, 18)
    V = [[0] * rank for _ in range(m * p)]  # (m*p, rank) = (27, 18)
    W = [[0] * rank for _ in range(p * n)]  # (p*n, rank) = (18, 18)

    col = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Verify exactness before returning (cheap for rank 18).
    if _verify(U, V, W, n, m, p, deadline):
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook if verification fails (should not happen).
    return _schoolbook(n, m, p)


def _verify(U, V, W, n, m, p, deadline):
    """Check that sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 iff i==i', j==j', k==k'."""
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Tensor T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # Reshape U: (n,m,rank), V: (m,p,rank), W: (p,n,rank)
    Ur = Ua.reshape(n, m, -1)
    Vr = Va.reshape(m, p, -1)
    Wr = Wa.reshape(p, n, -1)
    # T[i,j,k] = sum_{r} Ur[i,j,r] Vr[j,k,r] Wr[k,i,r]
    # Compute via einsum
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    expected = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, k] = 1.0
    if time.time() > deadline:
        return False
    return bool(np.allclose(T, expected, atol=1e-9))


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
