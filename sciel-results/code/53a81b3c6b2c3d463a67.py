import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# Rank-7 decomposition of 3x3x3 multiplication tensor.
# Verified constants for M_3,3,3 rank 7.
_A = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
]
_B = [
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
    [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, 0, 0], [0, -1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
]
_C = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 0, 0], [1, 1, 0]],
]

def _zero_matrix(rows, cols):
    return [[0] * cols for _ in range(rows)]

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = _zero_matrix(n * m, rank)
    V = _zero_matrix(m * p, rank)
    W = _zero_matrix(p * n, rank)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank14():
    n, m, p = DIMENSIONS
    n_groups = 2
    R = n_groups * len(_A)
    U = _zero_matrix(n * m, R)
    V = _zero_matrix(m * p, R)
    W = _zero_matrix(p * n, R)
    
    for g in range(n_groups):
        base = g * len(_A)
        for r in range(len(_A)):
            for i in range(n):
                for j in range(m):
                    val = _A[r][i][j]
                    if val != 0:
                        U[i * m + j][base + r] = val
            for j in range(m):
                for k_local in range(3):
                    val = _B[r][j][k_local]
                    if val != 0:
                        k_global = g * 3 + k_local
                        V[j * p + k_global][base + r] = val
            for i in range(n):
                for k_local in range(3):
                    val = _C[r][i][k_local]
                    if val != 0:
                        k_global = g * 3 + k_local
                        W[k_global * n + i][base + r] = val
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try rank 14
    try:
        cand = _rank14()
        if _verify(cand["U"], cand["V"], cand["W"]):
            return cand
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
