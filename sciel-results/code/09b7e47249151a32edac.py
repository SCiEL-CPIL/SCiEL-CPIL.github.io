"""Strassen-based rank-42 decomposition for 2x2 by 2x12 matrix multiplication."""
DIMENSIONS = [2, 2, 12]

import time

def construct():
    """Construct a rank-42 tensor decomposition using Strassen's algorithm.
    
    For multiplying (2x2) by (2x12), split the 12 columns into 6 groups of 2.
    For each group, apply Strassen's 7-multiplication formula for 2x2 blocks.
    Total rank = 6 * 7 = 42.
    
    Output W follows convention W[k*n+i, r] where n=2.
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # n=2, m=2, p=12

    # Number of 2-column blocks in the p dimension
    num_blocks = p // 2  # 6

    # Strassen coefficients for 2x2 matrix multiplication
    # m1 = (A00+A11)*(B00+B11)
    # m2 = (A10+A11)*B00
    # m3 = A00*(B01-B11)
    # m4 = A11*(B10-B00)
    # m5 = (A00+A01)*B11
    # m6 = (A10-A00)*(B00+B01)
    # m7 = (A01-A11)*(B10+B11)
    #
    # C00 = m1 - m4 + m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m3 + m2 + m6

    rank = num_blocks * 7  # 42

    # Allocate factor matrices
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42

    for g in range(num_blocks):
        base = g * 7
        # Column offsets for this block in V and W
        v_j0_k0 = 0 * p + 2 * g      # V row for (j=0, k=2g)
        v_j0_k1 = 0 * p + 2 * g + 1  # V row for (j=0, k=2g+1)
        v_j1_k0 = 1 * p + 2 * g      # V row for (j=1, k=2g)
        v_j1_k1 = 1 * p + 2 * g + 1  # V row for (j=1, k=2g+1)

        w_k0_i0 = (2 * g) * n + 0    # W row for (k=2g, i=0)
        w_k0_i1 = (2 * g) * n + 1    # W row for (k=2g, i=1)
        w_k1_i0 = (2 * g + 1) * n + 0  # W row for (k=2g+1, i=0)
        w_k1_i1 = (2 * g + 1) * n + 1  # W row for (k=2g+1, i=1)

        # r=0: m1 = (A00+A11)*(B00+B11)
        col = base + 0
        U[0][col] = 1   # A00
        U[3][col] = 1   # A11
        V[v_j0_k0][col] = 1  # B00
        V[v_j1_k1][col] = 1  # B11
        W[w_k0_i0][col] = 1  # C00
        W[w_k1_i1][col] = 1  # C11

        # r=1: m2 = (A10+A11)*B00
        col = base + 1
        U[2][col] = 1   # A10
        U[3][col] = 1   # A11
        V[v_j0_k0][col] = 1  # B00
        W[w_k0_i1][col] = 1  # C10
        W[w_k1_i1][col] = -1 # C11

        # r=2: m3 = A00*(B01-B11)
        col = base + 2
        U[0][col] = 1   # A00
        V[v_j0_k1][col] = 1   # B01
        V[v_j1_k1][col] = -1  # -B11
        W[w_k1_i0][col] = 1   # C01
        W[w_k1_i1][col] = 1   # C11

        # r=3: m4 = A11*(B10-B00)
        col = base + 3
        U[3][col] = 1   # A11
        V[v_j1_k0][col] = 1   # B10
        V[v_j0_k0][col] = -1  # -B00
        W[w_k0_i0][col] = 1   # C00
        W[w_k0_i1][col] = 1   # C10

        # r=4: m5 = (A00+A01)*B11
        col = base + 4
        U[0][col] = 1   # A00
        U[1][col] = 1   # A01
        V[v_j1_k1][col] = 1  # B11
        W[w_k0_i0][col] = -1 # C00
        W[w_k1_i0][col] = 1  # C01

        # r=5: m6 = (A10-A00)*(B00+B01)
        col = base + 5
        U[2][col] = 1    # A10
        U[0][col] = -1   # -A00
        V[v_j0_k0][col] = 1  # B00
        V[v_j0_k1][col] = 1  # B01
        W[w_k1_i1][col] = 1  # C11

        # r=6: m7 = (A01-A11)*(B10+B11)
        col = base + 6
        U[1][col] = 1    # A01
        U[3][col] = -1   # -A11
        V[v_j1_k0][col] = 1  # B10
        V[v_j1_k1][col] = 1  # B11
        W[w_k0_i0][col] = 1  # C00

    return {"U": U, "V": V, "W": W}
