"""Matrix multiplication 2,2,12 — rank-42 Strassen decomposition."""
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-42 exact tensor decomposition for 2x2 * 2x12 matrix mult.

    Uses 6 independent 2x2x2 Strassen blocks.
    """
    n, m, p = 2, 2, 12
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 7 products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # U_coeffs[s]: list of (ai, aj, coeff)
    U_coeffs = [
        [(0, 0, 1), (1, 1, 1)],           # M1
        [(1, 0, 1), (1, 1, 1)],           # M2
        [(0, 0, 1)],                      # M3
        [(1, 1, 1)],                      # M4
        [(0, 0, 1), (0, 1, 1)],           # M5
        [(1, 0, 1), (0, 0, -1)],          # M6
        [(0, 1, 1), (1, 1, -1)],          # M7
    ]
    
    # V_coeffs[s]: list of (bj, bk_rel, coeff)
    # bk_rel is 0 or 1 within the 2-column block
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],           # M1: B00+B11
        [(0, 0, 1)],                      # M2: B00
        [(0, 1, 1), (1, 1, -1)],          # M3: B01-B11
        [(1, 0, 1), (0, 0, -1)],          # M4: B10-B00
        [(1, 1, 1)],                      # M5: B11
        [(0, 0, 1), (0, 1, 1)],           # M6: B00+B01
        [(1, 0, 1), (1, 1, 1)],           # M7: B10+B11
    ]
    
    # W_coeffs[s]: list of (ci, ck_rel, coeff)
    # ck_rel is 0 or 1 within the 2-column block
    # W index is (k*n + i)
    W_coeffs = [
        [(0, 0, 1), (1, 0, 1)],           # M1 -> C00, C10
        [(1, 0, 1), (1, 0, -1)],          # M2 -> C10, -C10 (wait, C11? No)
        # Let's re-verify W mapping for Strassen
        # C00 = M1 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6
        
        # M1 contributes to C00 (col 0) and C11 (col 1)
        # M2 contributes to C10 (col 0) and C11 (col 1) with -1
        # M3 contributes to C01 (col 1)
        # M4 contributes to C10 (col 0)
        # M5 contributes to C00 (col 0) with -1 and C01 (col 1)
        # M6 contributes to C11 (col 1)
        # M7 contributes to C00 (col 0)
        
        # Correct W_coeffs:
    ]
    
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],           # M1 -> C00, C11
        [(1, 0, 1), (1, 1, -1)],          # M2 -> C10, -C11
        [(0, 1, 1)],                      # M3 -> C01
        [(1, 0, 1)],                      # M4 -> C10
        [(0, 0, -1), (0, 1, 1)],          # M5 -> -C00, C01
        [(1, 1, 1)],                      # M6 -> C11
        [(0, 0, 1)],                      # M7 -> C00
    ]

    col = 0
    for t in range(6):
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
