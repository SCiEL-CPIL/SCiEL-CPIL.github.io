import time

_DEADLINE = 115.0
_START = None


def construct():
    global _START
    if _START is None:
        _START = time.time()

    n, m, p = 2, 3, 9
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A_L (2x2) * B_T (2x9) using Strassen for 2x2x2 blocks
    # A indices: 0=A[0,0], 1=A[0,1], 3=A[1,0], 4=A[1,1]
    # B indices: j*p+k

    def strassen_block(k0, k1, col_start):
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1

        u_a = 0
        u_b = 1
        u_c = 3
        u_d = 4

        # M1 = (a+d)(e+h)
        U[u_a][col_start] += 1
        U[u_d][col_start] += 1
        V[v_e][col_start] += 1
        V[v_h][col_start] += 1

        # M2 = (c+d)e
        U[u_c][col_start+1] += 1
        U[u_d][col_start+1] += 1
        V[v_e][col_start+1] += 1

        # M3 = a(f-h)
        U[u_a][col_start+2] += 1
        V[v_f][col_start+2] += 1
        V[v_h][col_start+2] -= 1

        # M4 = d(g-e)
        U[u_d][col_start+3] += 1
        V[v_g][col_start+3] += 1
        V[v_e][col_start+3] -= 1

        # M5 = (a+b)h
        U[u_a][col_start+4] += 1
        U[u_b][col_start+4] += 1
        V[v_h][col_start+4] += 1

        # M6 = (c-a)(e+f)
        U[u_c][col_start+5] += 1
        U[u_a][col_start+5] -= 1
        V[v_e][col_start+5] += 1
        V[v_f][col_start+5] += 1

        # M7 = (b-d)(g+h)
        U[u_b][col_start+6] += 1
        U[u_d][col_start+6] -= 1
        V[v_g][col_start+6] += 1
        V[v_h][col_start+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        W[w_00][col_start] += 1
        W[w_00][col_start+3] += 1
        W[w_00][col_start+4] -= 1
        W[w_00][col_start+6] += 1

        W[w_01][col_start+2] += 1
        W[w_01][col_start+4] += 1

        W[w_10][col_start+1] += 1
        W[w_10][col_start+3] += 1

        W[w_11][col_start] += 1
        W[w_11][col_start+1] -= 1
        W[w_11][col_start+2] += 1
        W[w_11][col_start+5] += 1

    # 4 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        strassen_block(k0, k1, col)
        col += 7

    # 2x2x1 block for column 8 (rank 4)
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    col += 4

    # Part 2: A_R (2x1) * B_B (1x9), rank 18
    u_ar0 = 2  # A[0,2]
    u_ar1 = 5  # A[1,2]

    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1

        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2

    return {"U": U, "V": V, "W": W}
