import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for 150s limit
    
    n, m, p = DIMENSIONS
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        U[a][col] += 1; U[d][col] += 1; V[e][col] += 1; V[h][col] += 1
        W[c00][col] += 1; W[c11][col] += 1; col += 1
        U[c][col] += 1; U[d][col] += 1; V[e][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1; col += 1
        U[a][col] += 1; V[f][col] += 1; V[h][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1; col += 1
        U[d][col] += 1; V[g][col] += 1; V[e][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1; col += 1
        U[a][col] += 1; U[b][col] += 1; V[h][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1; col += 1
        U[c][col] += 1; U[a][col] -= 1; V[e][col] += 1; V[f][col] += 1
        W[c11][col] += 1; col += 1
        U[b][col] += 1; U[d][col] -= 1; V[g][col] += 1; V[h][col] += 1
        W[c00][col] += 1; col += 1
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        if col >= rank:
            return False
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        U[a][col] += 1; V[e][col] += 1; W[c00][col] += 1; col += 1
        U[b][col] += 1; V[g][col] += 1; W[c00][col] += 1; col += 1
        U[c][col] += 1; V[e][col] += 1; W[c10][col] += 1; col += 1
        U[d][col] += 1; V[g][col] += 1; W[c10][col] += 1; col += 1
        return True

    # A_left @ B_top: i0=0, j0=0
    for k0 in [0, 2, 4]:
        add_strassen_222(0, 0, k0)
    add_schoolbook_221(0, 0, 6)
    
    # A_right @ B_bot: i0=0, j0=2
    for k0 in [0, 2, 4]:
        add_strassen_222(0, 2, k0)
    add_schoolbook_221(0, 2, 6)
    
    return {"U": U, "V": V, "W": W}
