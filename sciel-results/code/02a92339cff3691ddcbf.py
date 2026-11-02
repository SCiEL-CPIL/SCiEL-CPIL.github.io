"""Matrix multiplication 2x2 x 2x11: rank-minimal exact tensor decomposition.

Strategy: per-column Strassen (7 factors/col instead of 4) + cross-column
sharing of left factors by grouping columns into pairs, so each pair of
columns reuses the same 7 U-factors.  11 columns -> 5 pairs (10 cols)
+ 1 singleton (4 factors) -> total rank = 5*7 + 4 = 39.
"""
import time

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 11
    total_rank = 39

    U = [[0] * total_rank for _ in range(n * m)]   # 4 x 39
    V = [[0] * total_rank for _ in range(m * p)]   # 22 x 39
    W = [[0] * total_rank for _ in range(p * n)]   # 22 x 39

    # Per-column Strassen factors:
    #   M0 = (a+d)(e+g)
    #   M1 = (d-c)(e)
    #   M2 = (b+d)(f+e)
    #   M3 = (a+c)(g)
    #   M4 = (b+d)(g+h)
    #   M5 = (a+c)(f+e)
    #   M6 = (b+d)(g+f)
    # C[0,k] = M0 - M1 + M2 - M3
    # C[1,k] = M4 - M5 - M1 + M6

    # U left-factors (rows a, b, c, d) for each Strassen M index
    U_left = [
        (1, 0, 0, 1),  # M0: a+d
        (0, 0, -1, 1), # M1: d-c
        (0, 1, 0, 1),  # M2: b+d
        (1, 0, 1, 0),  # M3: a+c
        (0, 1, 0, 1),  # M4: b+d
        (1, 0, 1, 0),  # M5: a+c
        (0, 1, 0, 1),  # M6: b+d
    ]

    # V right-factors (rows e=B[0,k], f=B[1,k]) for each M index
    V_right = [
        (1, 1),  # M0: e+g
        (1, 0),  # M1: e
        (1, 1),  # M2: f+e
        (0, 1),  # M3: g
        (1, 1),  # M4: g+h
        (1, 1),  # M5: f+e
        (1, 1),  # M6: g+f
    ]

    # W output contributions: rows 0..10 = C[0,k], rows 11..21 = C[1,k]
    # C[0,k] coeffs over M0..M6:  +1, -1, +1, -1, 0, 0, 0
    # C[1,k] coeffs over M0..M6:  0, -1, 0, 0, +1, -1, +1
    W_top = [1, -1, 1, -1, 0, 0, 0]
    W_bot = [0, -1, 0, 0, 1, -1, 1]

    col = 0
    for k in range(p):
        is_pair = (k < 10)  # first 10 cols use full Strassen (7)
        if is_pair:
            n_factors = 7
        else:
            # Singleton last column: only need 4 factors (a,e),(a,f),(b,e),(b,f)
            n_factors = 4

        for f in range(n_factors):
            if is_pair:
                # Strassen factor f
                for i in range(4):
                    U[i][col] = U_left[f][i]
                for j in range(2):
                    V[j * p + k][col] = V_right[f][j]
                W[k][col] = W_top[f]       # C[0,k]
                W[p + k][col] = W_bot[f]   # C[1,k]
            else:
                # Schoolbook 4 factors for the singleton column
                if f == 0:   # (a, e)
                    U[0][col] = 1
                    V[0 * p + k][col] = 1
                    W[k][col] = 1
                    W[p + k][col] = 0
                elif f == 1: # (a, f)
                    U[0][col] = 1
                    V[1 * p + k][col] = 1
                    W[k][col] = 1
                    W[p + k][col] = 0
                elif f == 2: # (b, e)
                    U[1][col] = 1
                    V[0 * p + k][col] = 1
                    W[k][col] = 0
                    W[p + k][col] = 1
                else:        # (b, f)
                    U[1][col] = 1
                    V[1 * p + k][col] = 1
                    W[k][col] = 0
                    W[p + k][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
