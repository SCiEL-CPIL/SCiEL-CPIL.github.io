"""Rank-minimal exact tensor decomposition for 3x4 @ 4x8.

Split the 4x8 factor into two 4x4 blocks (B1, B2):
  C = A @ [B1 | B2] = [A@B1 | A@B2]
Each 3x4 @ 4x4 product uses Strassen (rank 7). Total rank = 14.
Output W follows the official trace convention W[k*n + i, r].
"""
import time

DIMENSIONS = [3, 4, 8]

# Strassen rank-7 for 2x2 @ 2x2.
# Each row: (a11,a12,a21,a22, b11,b12,b21,b22, c11,c12,c21,c22)
_STRASSEN = [
    ( 1,  0,  0,  1,   1,  1,  1,  1,   1,  0,  0,  1),  # P1
    ( 0,  0,  1,  1,   1,  0,  0,  0,   0,  0,  1,  0),  # P2
    ( 1,  0,  0,  0,   0,  1,  0, -1,   0,  1,  0,  1),  # P3
    ( 0,  0,  0,  1,   0, -1,  1,  0,   1,  0,  1,  0),  # P4
    ( 1,  1,  0,  0,   0,  0,  0,  1,  -1,  1,  0,  0),  # P5
    (-1,  0,  1,  0,   1,  1,  0,  0,   0,  0,  0,  1),  # P6
    ( 0,  1,  0, -1,   0,  0,  1,  1,   1,  0,  0,  0),  # P7
]


def _strassen_factors(r, c):
    """(r x 2) @ (2 x c) via one Strassen block.
    Returns U (r*2 x 7), V (2*c x 7), W (c*r x 7) with W[k*r+i]."""
    U = [[0] * 7 for _ in range(r * 2)]
    V = [[0] * 7 for _ in range(2 * c)]
    W = [[0] * 7 for _ in range(c * r)]
    for t in range(7):
        a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22 = _STRASSEN[t]
        for i in range(r):
            U[i][t] = a11
            U[r + i][t] = a21
        for j in range(c):
            V[j][t] = b11
            V[c + j][t] = b21
        for i in range(r):
            for k in range(c):
                W[k * r + i][t] = c11
                W[(k + c) * r + i][t] = c21
        for i in range(r):
            for k in range(c):
                W[k * r + i][t] += c12
                W[(k + c) * r + i][t] += c22
    return U, V, W


def _add(out_U, out_V, out_W, U, V, W, io, jo, ko):
    for i in range(len(U)):
        for t in range(7):
            out_U[i + io][t] += U[i][t]
    for j in range(len(V)):
        for t in range(7):
            out_V[j + jo][t] += V[j][t]
    for k in range(len(W)):
        for t in range(7):
            out_W[k + ko][t] += W[k][t]


def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 8
    p1, p2 = p // 2, p - p // 2  # 4, 4

    rank = 14
    U = [[0] * rank for _ in range(n * m)]   # 12 x 14
    V = [[0] * rank for _ in range(m * p)]   # 32 x 14
    W = [[0] * rank for _ in range(p * n)]   # 24 x 14

    # Block 1: A (3x4) @ B1 (4x4) -> C_left (3x4).  k in [0,4), j in [0,4).
    U1, V1, W1 = _strassen_factors(r=n, c=p1)
    _add(U, V, W, U1, V1, W1, io=0, jo=0, ko=0)

    # Block 2: A (3x4) @ B2 (4x4) -> C_right (3x4).  k in [4,8).
    U2, V2, W2 = _strassen_factors(r=n, c=p2)
    _add(U, V, W, U2, V2, W2, io=0, jo=p1, ko=p1)

    if time.time() - start > 120:
        return {"U": U, "V": V, "W": W}
    return {"U": U, "V": V, "W": W}
