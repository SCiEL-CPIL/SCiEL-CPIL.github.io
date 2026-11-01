"""Matrix multiplication 2,2,12 — rank-42 Strassen decomposition."""
import time
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-42 exact tensor decomposition for 2x2 * 2x12 matrix mult.

    Conventions:
      U: shape (n*m, rank) = (4, 42);  U[i*m+j, r] multiplies A[i][j]
      V: shape (m*p, rank) = (24, 42); V[j*p+k, r] multiplies B[j][k]
      W: shape (p*n, rank) = (24, 42); W[k*n+i, r] contributes to C[i][k] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n, m, p = 2, 2, 12
    num_pairs = p // 2  # 6
    rank = num_pairs * 7  # 42

    U = [[0] * rank for _ in range(n * m)]    # 4 x 42
    V = [[0] * rank for _ in range(m * p)]    # 24 x 42
    W = [[0] * rank for _ in range(p * n)]    # 24 x 42

    # Strassen 7 products for 2x2 block:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    #
    # Output reconstruction:
    #   C00 = M1 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M6

    # For each product s: list of (input_index, coeff) for U, V, W
    # U indices: (ai, aj) -> row ai*m + aj
    # V indices: (bj, bk_rel) -> row bj*p + (base_col + bk_rel)
    # W indices: (ci, ck_rel) -> row (base_col + ck_rel)*n + ci

    U_coeffs = [
        [(0, 0, 1), (1, 1, 1)],           # M1
        [(1, 0, 1), (1, 1, 1)],           # M2
        [(0, 0, 1)],                      # M3
        [(1, 1, 1)],                      # M4
        [(0, 0, 1), (0, 1, 1)],           # M5
        [(1, 0, 1), (0, 0, -1)],          # M6
        [(0, 1, 1), (1, 1, -1)],          # M7
    ]
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],           # M1
        [(0, 0, 1)],                      # M2
        [(0, 1, 1), (1, 1, -1)],          # M3
        [(1, 0, 1), (0, 0, -1)],          # M4
        [(1, 1, 1)],                      # M5
        [(0, 0, 1), (0, 1, 1)],           # M6
        [(1, 0, 1), (1, 1, 1)],           # M7
    ]
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],           # M1 -> C00 += 1, C11 += 1
        [(1, 0, 1), (1, 1, -1)],          # M2 -> C10 += 1, C11 += -1
        [(0, 1, 1), (1, 1, 1)],           # M3 -> C01 += 1, C11 += 1
        [(0, 0, 1), (1, 0, 1)],           # M4 -> C00 += 1, C10 += 1
        [(0, 0, -1), (0, 1, 1)],          # M5 -> C00 += -1, C01 += 1
        [(1, 1, 1)],                      # M6 -> C11 += 1
        [(0, 0, 1)],                      # M7 -> C00 += 1
    ]

    col = 0
    for t in range(num_pairs):
        if time.time() > deadline:
            break
        base_col = 2 * t
        for s in range(7):
            for (ai, aj, c) in U_coeffs[s]:
                U[ai * m + aj][col] = c
            for (bj, bk_rel, c) in V_coeffs[s]:
                V[bj * p + base_col + bk_rel][col] = c
            for (ci, ck_rel, c) in W_coeffs[s]:
                W[(base_col + ck_rel) * n + ci][col] = c
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
