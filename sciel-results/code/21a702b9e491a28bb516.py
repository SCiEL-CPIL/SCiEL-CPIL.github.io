"""Slice-rank decomposition for M_{3,4,7} using explicit linear algebra."""

import numpy as np

DIMENSIONS = [3, 4, 7]


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False, (i, j, k, s)
    return True, None


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    total_U_rows = n * m  # 12
    total_V_rows = m * p  # 28
    total_W_rows = p * n  # 21

    # Schoolbook decomposition: rank = n*m*p = 84
    # Each term (i,j,k) contributes one rank-1 component
    rank = n * m * p
    U = np.zeros((total_U_rows, rank))
    V = np.zeros((total_V_rows, rank))
    W = np.zeros((total_W_rows, rank))

    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1

    # Verify the decomposition
    valid, err = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        raise ValueError(f"Decomposition failed at {err}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
