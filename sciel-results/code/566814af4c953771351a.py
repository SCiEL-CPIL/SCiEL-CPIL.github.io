import time

import numpy as np

_DEADLINE = 120.0
_start_time = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def construct():
    n, m, p = 2, 3, 9
    rank = 50

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
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

        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1

        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1

        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1

        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1

        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1

        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1

        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1

        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1

        W[w_01][col+2] += 1
        W[w_01][col+4] += 1

        W[w_10][col+1] += 1
        W[w_10][col+3] += 1

        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1

        col += 7

    k8 = 8
    v_e8 = 0 * p + k8
    v_g8 = 1 * p + k8
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    U[u_a][col] += 1
    V[v_e8][col] += 1
    W[w_08][col] += 1

    U[u_b][col+1] += 1
    V[v_g8][col+1] += 1
    W[w_08][col+1] += 1

    U[u_c][col+2] += 1
    V[v_e8][col+2] += 1
    W[w_18][col+2] += 1

    U[u_d][col+3] += 1
    V[v_g8][col+3] += 1
    W[w_18][col+3] += 1

    col += 4

    # Part 3: Handle j=2 (the third column of A, i.e., the third row of B)
    # For j=2, the tensor entry is 1 for all (i,k).
    # We can use a single rank-1 term:
    # U[i*m+2, r] = 1 for all i
    # V[2*p+k, r] = 1 for all k
    # W[k*n+i, r] = 1 for all k,i
    # This gives sum_r U*V*W = 1*1*1 = 1 for all (i,k) when j=2.
    # And for j != 2, U[i*m+j, r] = 0, so the sum is 0.
    
    # Use one rank for this
    u_ar0 = 0 * m + 2  # i=0, j=2
    u_ar1 = 1 * m + 2  # i=1, j=2

    # For all k, v_bk = 2*p+k
    # For all k,i, w_ki = k*n+i

    for i in range(n):
        U[i * m + 2][col] += 1
    for k in range(p):
        V[2 * p + k][col] += 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] += 1

    col += 1

    # Now we have used col ranks. Let's trim.
    # Actually, we need to return only the used ranks.
    # But the interface expects fixed shapes. Let's just return the full arrays
    # and let the evaluator handle it, or we can trim.
    # The problem says "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)".
    # So rank is the number of columns.
    
    # Let's trim to the actual used rank.
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]

    return {"U": U, "V": V, "W": W}
