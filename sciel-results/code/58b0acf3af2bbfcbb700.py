"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
DEADLINE_START = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 3.0


def _time_left():
    return (DEADLINE_START + TIME_LIMIT - SAFETY_MARGIN) - time.time()


def _schoolbook_columns():
    n, m, p = 2, 3, 5
    cols = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                cols.append((i, j, k))
    return cols


def _to_matrices(cols):
    n, m, p = 2, 3, 5
    rank = len(cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (i, j, k) in enumerate(cols):
        U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1
    return {"U": U, "V": V, "W": W}, rank


def _verify(U, V, W):
    n, m, p = 2, 3, 5
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == j == k:
                    continue
                if (i, j, k) in [(i, j, k)]:
                    pass
    # check all non-(matching) entries are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                want = 1 if (i < n and j < m and k < p and i == j == k) else 0
                if i >= n or j >= m or k >= p:
                    continue
    # proper tensor entry check: T_{i,j,k}=1 for the matching product term i in [n], j in [m], k in [p] is 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                want = 1 if (i == j and j == k and i < n and j < m) else 0
                if s != want:
                    return False
    return True


def construct():
    best_rank = 999999
    best = None
    # Candidate 1: schoolbook (guaranteed correct).
    cols = _schoolbook_columns()
    res, rank = _to_matrices(cols)
    if _verify(res["U"], res["V"], res["W"]) and rank < best_rank:
        best_rank = rank
        best = res
    return best if best is not None else res
# EVOLVE-BLOCK-END
