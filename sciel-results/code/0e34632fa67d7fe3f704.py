import time
import numpy as np

_DEADLINE = 115.0
_START = None


def construct():
    global _START
    if _START is None:
        _START = time.time()

    n, m, p = 2, 3, 9

    # We will build a rank-49 decomposition.
    # Structure:
    # - 3 Strassen blocks for column pairs (0,1), (2,3), (4,5): 3 * 7 = 21 terms
    # - 1 custom block for columns 6,7,8 (j=0,1 only): 10 terms
    # - Direct loop for j=2, all 9 columns: 18 terms
    # Total: 21 + 10 + 18 = 49

    RANK = 49
    U = [[0] * RANK for _ in range(n * m)]
    V = [[0] * RANK for _ in range(m * p)]
    W = [[0] * RANK for _ in range(p * n)]

    col = 0

    # Strassen block for 2x2x2 (columns k0, k1)
    def strassen_block(k0, k1, col_start):
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1

        u_a = 0  # A_{0,0}
        u_b = 1  # A_{1,0}
        u_c = 3  # A_{0,1}
        u_d = 4  # A_{1,1}

        # M1 = (a+d)(e+h)
        U[u_a][col_start] += 1
        U[u_d][col_start] += 1
        V[v_e][col_start] += 1
        V[v_h][col_start] += 1

        # M2 = (c+d)e
        U[u_c][col_start + 1] += 1
        U[u_d][col_start + 1] += 1
        V[v_e][col_start + 1] += 1

        # M3 = a(f-h)
        U[u_a][col_start + 2] += 1
        V[v_f][col_start + 2] += 1
        V[v_h][col_start + 2] -= 1

        # M4 = d(g-e)
        U[u_d][col_start + 3] += 1
        V[v_g][col_start + 3] += 1
        V[v_e][col_start + 3] -= 1

        # M5 = (a+b)h
        U[u_a][col_start + 4] += 1
        U[u_b][col_start + 4] += 1
        V[v_h][col_start + 4] += 1

        # M6 = (c-a)(e+f)
        U[u_c][col_start + 5] += 1
        U[u_a][col_start + 5] -= 1
        V[v_e][col_start + 5] += 1
        V[v_f][col_start + 5] += 1

        # M7 = (b-d)(g+h)
        U[u_b][col_start + 6] += 1
        U[u_d][col_start + 6] -= 1
        V[v_g][col_start + 6] += 1
        V[v_h][col_start + 6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # C00 = M1 + M4 - M5 + M7
        W[w_00][col_start] += 1
        W[w_00][col_start + 3] += 1
        W[w_00][col_start + 4] -= 1
        W[w_00][col_start + 6] += 1

        # C01 = M3 + M5
        W[w_01][col_start + 2] += 1
        W[w_01][col_start + 4] += 1

        # C10 = M2 + M4
        W[w_10][col_start + 1] += 1
        W[w_10][col_start + 3] += 1

        # C11 = M1 - M2 + M3 + M6
        W[w_11][col_start] += 1
        W[w_11][col_start + 1] -= 1
        W[w_11][col_start + 2] += 1
        W[w_11][col_start + 5] += 1

    # 3 Strassen blocks for column pairs (0,1), (2,3), (4,5)
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        strassen_block(k0, k1, col)
        col += 7

    # Custom block for columns 6, 7, 8 (j=0,1 only)
    # We need to compute:
    # C_{0,k} = A_{0,0} * B_{0,k} + A_{0,1} * B_{1,k} for k in {6,7,8}
    # C_{1,k} = A_{1,0} * B_{0,k} + A_{1,1} * B_{1,k} for k in {6,7,8}
    # That's 6 outputs, each with 2 terms. Direct rank would be 12.
    # Let's try to find a rank 10 algorithm.
    # We can use a combination of Strassen-like ideas.
    # Let's define:
    # M1 = (A_{0,0} + A_{1,0}) * (B_{0,6} + B_{0,7} + B_{0,8})
    # M2 = A_{0,1} * (B_{1,6} + B_{1,7} + B_{1,8})
    # M3 = A_{1,1} * (B_{1,6} + B_{1,7} + B_{1,8})
    # This gives:
    # C_{0,6} = A_{0,0}*B_{0,6} + A_{0,1}*B_{1,6} = (M1 term) + (M2 term)
    # But M1 has A_{1,0}*B_{0,6} which is not in C_{0,6}.
    # So this doesn't work directly.
    
    # Let's use a different approach.
    # We can use 2 Strassen blocks for (6,7) and (7,8) and share some terms?
    # Or just use rank 12 for simplicity?
    # If we use rank 12, total is 21 + 12 + 18 = 51. Worse than 50.
    
    # Let's stick to rank 50 for now, but with the correct structure.
    # Actually, let's just use the original rank 50 solution but ensure it's correct.
    # The original solution had 4 Strassen blocks (28) + 1 single (4) + 1 loop (18) = 50.
    # Let's rebuild that.
    
    # Reset
    U = [[0] * 50 for _ in range(n * m)]
    V = [[0] * 50 for _ in range(m * p)]
    W = [[0] * 50 for _ in range(p * n)]
    col = 0

    # 4 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        strassen_block(k0, k1, col)
        col += 7

    # Single block for column 8 (j=0,1)
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    # M1 = A_{0,0} * B_{0,8}
    U[u_a][col] += 1
    V[v_e][col] += 1

    # M2 = A_{1,0} * B_{0,8}
    U[u_b][col + 1] += 1
    V[v_g][col + 1] += 1

    # M3 = A_{0,1} * B_{0,8}
    U[u_c][col + 2] += 1
    V[v_e][col + 2] += 1

    # M4 = A_{1,1} * B_{0,8}
    U[u_d][col + 3] += 1
    V[v_g][col + 3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    W[w_08][col] += 1
    W[w_08][col + 1] += 1
    W[w_18][col + 2] += 1
    W[w_18][col + 3] += 1

    col += 4

    # Direct loop for j=2, all 9 columns
    for k in range(9):
        v_b = 2 * p + k
        U[2][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1

        U[5][col + 1] += 1
        V[v_b][col + 1] += 1
        W[k * n + 1][col + 1] += 1

        col += 2

    return {"U": U, "V": V, "W": W}
