import time

import numpy as np

_DEADLINE = 120.0  # wall-clock budget (s); evaluator kills at 150 s

def _time_left():
    # simple deadline helper: returns seconds remaining
    return _DEADLINE - (time.time() - time.time())

def construct():
    # Dimensions: n=2 (rows of A), m=3 (inner), p=9 (cols of B)
    n, m, p = 2, 3, 9

    # Rank 50 explicit construction:
    #  4 Strassen blocks over B columns (0,1),(2,3),(4,5),(6,7):  4*7 = 28 terms
    #  1 naive 2x1x2x1 block   over B column 8:                   4      terms
    #  9 + 9 naive single-product terms for A_right * B_bottom:   18     terms
    # ---------------------------------------------------------------------------
    # Total rank = 50
    rank = 50

    # U has shape (n*m, rank) = (6, 50)
    # V has shape (m*p, rank) = (27, 50)
    # W has shape (p*n, rank) = (18, 50)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: Strassen-style 2x2 blocks over A_top (2x2) and B_top (2x9)
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1

        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1 = (a + d)(e + h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1

        # M2 = (c + d) * e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1

        # M3 = a * (f - h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1

        # M4 = d * (g - e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1

        # M5 = (a + b) * h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1

        # M6 = (c - a)(e + f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1

        # M7 = (b - d)(g + h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1

        # Reconstruct
        W[w_00][col]   += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1

        W[w_01][col+2] += 1
        W[w_01][col+4] += 1

        W[w_10][col+1] += 1
        W[w_10][col+3] += 1

        W[w_11][col]   += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1

        col += 7

    # Part 2: Naive 2x1 block for last B column k = 8
    k8 = 8
    v_e8 = 0 * p + k8
    v_g8 = 1 * p + k8

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    # P1 = a * B_{0,8}
    U[u_a][col]   += 1
    V[v_e8][col]  += 1
    W[w_08][col]  += 1

    # P2 = b * B_{1,8}
    U[u_b][col+1] += 1
    V[v_g8][col+1] += 1
    W[w_08][col+1] += 1

    # P3 = c * B_{0,8}
    U[u_c][col+2] += 1
    V[v_e8][col+2] += 1
    W[w_18][col+2] += 1

    # P4 = d * B_{1,8}
    U[u_d][col+3] += 1
    V[v_g8][col+3] += 1
    W[w_18][col+3] += 1

    col += 4

    # Part 3: A_right (col 2) x B_bottom (row 2)
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2

    for k in range(p):
        v_bk = 2 * p + k
        w_0k = k * n + 0
        w_1k = k * n + 1

        U[u_ar0][col]   += 1
        V[v_bk][col]    += 1
        W[w_0k][col]    += 1

        U[u_ar1][col+1] += 1
        V[v_bk][col+1]  += 1
        W[w_1k][col+1]  += 1

        col += 2

    return {"U": U, "V": V, "W": W}
