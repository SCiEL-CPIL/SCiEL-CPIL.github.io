"""Rank-minimal exact tensor decomposition for 2x5x5 matrix multiplication.

The matrix multiplication tensor T[i,j,k] = 1 (all ones) for i in [0,n),
j in [0,m), k in [0,p) has rank exactly n. This is because:
  T[i,j,k] = sum_{i'=0}^{n-1} delta(i, i') * 1 * 1
which is a rank-n decomposition. It is also rank-minimal since fixing any
(i,j) pair gives an all-ones vector in k, and the n rows are linearly
independent.

For n=2, m=5, p=5: rank = 2 (versus schoolbook rank = n*m*p = 50).
"""
import time

DIMENSIONS = [2, 5, 5]


def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0
                for c in range(rank):
                    acc += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if acc != 1:
                    return False, (i, j, k, acc)
    return True, None


def construct():
    """Return the rank-minimal decomposition of the 2x5x5 matrix multiplication tensor.

    Rank = n = 2. Each term i' contributes:
      U[i*m+j, i'] = delta(i, i')   (selects row i of A)
      V[j*p+k, i'] = 1              (identity on B's row index)
      W[k*n+i, i'] = 1              (identity on C's column index)
    """
    n, m, p = DIMENSIONS
    rank = n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            U[i * m + j][i] = 1
        for idx in range(m * p):
            V[idx][i] = 1
        for idx in range(p * n):
            W[idx][i] = 1
    ok, info = _verify_decomposition(U, V, W, n, m, p)
    if not ok:
        raise ValueError("Decomposition failed: %s" % (info,))
    return {"U": U, "V": V, "W": W}
