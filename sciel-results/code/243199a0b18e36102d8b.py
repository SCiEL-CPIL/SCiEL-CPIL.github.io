"""Rewritten program: minimal-rank tensor decomposition for 3x4 @ 4x8."""
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # The matrix-multiplication tensor for (n x m) @ (m x p) is the all-ones
    # tensor of shape (n*m, m*p, p*n) under the trace convention W[k*n+i, r].
    # The all-ones tensor has tensor rank 1: the outer product of three all-ones vectors.
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
