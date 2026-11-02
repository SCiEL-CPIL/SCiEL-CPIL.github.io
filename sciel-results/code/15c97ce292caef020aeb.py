"""Block-Strassen rank-189 exact tensor decomposition for 6x6 matrix multiplication."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = 6, 6, 6

    # Strassen's 7 terms for 2x2 block multiplication
    # Blocks are 3x3 submatrices
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each Strassen term s, we need linear combinations of blocks for A and B
    # alpha[s][u] = coefficient of block A_u in X_s
    # beta[s][v] = coefficient of block B_v in Y_s
    # gamma[s][w] = coefficient of result block M_s in C_w

    # Block indices: 00=0, 01=1, 10=2, 11=3
    # alpha: for each s, coefficients for A00, A01, A10, A11
    # beta: for each s, coefficients for B00, B01, B10, B11
    # gamma: for each s, coefficients for C00, C01, C10, C11

    alpha = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]

    beta = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [0, 0, 1, -1],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1],   # M7: B10 + B11
    ]

    gamma = [
        [1, 0, 0, 0],   # M1 -> C00
        [0, 0, 1, 0],   # M2 -> C10
        [0, 1, 0, 0],   # M3 -> C01
        [1, 0, 1, 0],   # M4 -> C00, C10
        [-1, 0, 0, 0],  # M5 -> C00
        [0, 0, 0, 1],   # M6 -> C11
        [1, 0, 0, 1],   # M7 -> C00, C11
    ]

    # Each Strassen term involves multiplying two 3x3 matrices (schoolbook: 27 terms)
    # Total rank = 7 * 27 = 189

    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189

    col = 0
    for s in range(7):
        for a in range(3):
            for b in range(3):
                for d in range(3):
                    # Schoolbook term: X[a,b] * Y[b,d] -> M[a,d]
                    # For this term, U[i,j] gets contribution from X[a,b]
                    # V[j,k] gets contribution from Y[b,d]
                    # W[k,i] gets contribution to M[a,d]

                    # X_s[a,b] = sum_u alpha[s][u] * A_u[a,b]
                    # Y_s[b,d] = sum_v beta[s][v] * B_v[b,d]
                    # M_s[a,d] contributes to C_w with coeff gamma[s][w]

                    # For each block u of A, find global indices for local (a,b)
                    for u in range(4):
                        if alpha[s][u] != 0:
                            # Block u: u=0 -> rows 0-2, cols 0-2; u=1 -> rows 0-2, cols 3-5
                            # u=2 -> rows 3-5, cols 0-2; u=3 -> rows 3-5, cols 3-5
                            row_base = (u // 2) * 3
                            col_base = (u % 2) * 3
                            gi = row_base + a  # global row i
                            gj = col_base + b  # global col j
                            U[gi * m + gj][col] = alpha[s][u]

                    for v in range(4):
                        if beta[s][v] != 0:
                            # Block v of B: v=0 -> rows 0-2, cols 0-2; etc.
                            row_base = (v // 2) * 3
                            col_base = (v % 2) * 3
                            gj2 = row_base + b  # global row j (from B's row)
                            gk = col_base + d   # global col k
                            V[gj2 * p + gk][col] = beta[s][v]

                    for w in range(4):
                        if gamma[s][w] != 0:
                            # M_s[a,d] contributes to C_w
                            # C_w: w=0 -> rows 0-2, cols 0-2; w=1 -> rows 0-2, cols 3-5
                            # w=2 -> rows 3-5, cols 0-2; w=3 -> rows 3-5, cols 3-5
                            row_base = (w // 2) * 3
                            col_base = (w % 2) * 3
                            gi2 = row_base + a  # global row i
                            gk2 = col_base + d  # global col k
                            W[gk2 * n + gi2][col] = gamma[s][w]

                    col += 1

    if time.time() > deadline:
        # Return what we have (shouldn't happen, but safety)
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
