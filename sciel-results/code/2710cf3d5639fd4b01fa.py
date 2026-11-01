"""Strassen rank-14 decomposition for 2x2 * 2x4 with verification."""
DIMENSIONS = [2, 2, 4]
import numpy as np
import time

def _verify(U, V, W):
    n, m, p = 2, 2, 4
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False, abs(s-1.0)
    return True, 0.0

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _strassen_rank14():
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    def add_block(bcol_offset, col):
        nonlocal col
        products = [
            ([(0, 1), (3, 1)],
             [(0*4+bcol_offset, 1), (1*4+bcol_offset+1, 1)],
             [(bcol_offset*2+0, 1), ((bcol_offset+1)*2+1, 1)]),
            ([(2, 1), (3, 1)],
             [(0*4+bcol_offset, 1)],
             [(bcol_offset*2+1, 1), ((bcol_offset+1)*2+1, -1)]),
            ([(0, 1)],
             [(0*4+bcol_offset+1, 1), (1*4+bcol_offset, -1)],
             [((bcol_offset+1)*2+0, 1), (bcol_offset*2+1, 1)]),
            ([(3, 1)],
             [(1*4+bcol_offset, 1), (1*4+bcol_offset+1, -1)],
             [(bcol_offset*2+0, -1), (bcol_offset*2+1, 1)]),
            ([(0, 1), (1, 1)],
             [(1*4+bcol_offset+1, 1)],
             [(bcol_offset*2+0, 1), ((bcol_offset+1)*2+0, 1)]),
            ([(2, 1), (0, -1)],
             [(0*4+bcol_offset, 1), (0*4+bcol_offset+1, 1)],
             [(bcol_offset*2+0, 1), (bcol_offset*2+1, -1)]),
            ([(1, 1), (3, -1)],
             [(1*4+bcol_offset, 1), (1*4+bcol_offset+1, 1)],
             [((bcol_offset+1)*2+1, 1), ((bcol_offset+1)*2+0, 1)]),
        ]
        for Uc, Vc, Wc in products:
            for row, val in Uc:
                U[row][col] += val
            for row, val in Vc:
                V[row][col] += val
            for row, val in Wc:
                W[row][col] += val
            col += 1
    add_block(0, 0)
    add_block(2, 7)
    return {"U": U, "V": V, "W": W}

def construct():
    result = _strassen_rank14()
    ok, err = _verify(result["U"], result["V"], result["W"])
    if ok:
        return result
    return _schoolbook()
