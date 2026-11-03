"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 42: 2 blocks of A (2x2) x 3 blocks of B cols (2x2) x 7 Strassen terms
    rank = 42

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for U (A matrix)
    # List of lists, each inner list contains (i_loc, j_loc, coeff) for term s
    U_coeffs = [
        [(0, 0, 1), (1, 1, 1)],       # M1: A00 + A11
        [(1, 0, 1), (1, 1, 1)],       # M2: A10 + A11
        [(0, 0, 1)],                  # M3: A00
        [(1, 1, 1)],                  # M4: A11
        [(0, 0, 1), (0, 1, 1)],       # M5: A00 + A01
        [(1, 0, 1), (0, 0, -1)],      # M6: A10 - A00
        [(0, 1, 1), (1, 1, -1)]       # M7: A01 - A11
    ]

    # Strassen coefficients for V (B matrix)
    # List of lists, each inner list contains (j_loc, k_loc, coeff) for term s
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],       # M1: B00 + B11
        [(0, 0, 1)],                  # M2: B00
        [(0, 1, 1), (1, 1, -1)],      # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],      # M4: B10 - B00
        [(1, 1, 1)],                  # M5: B11
        [(0, 0, 1), (0, 1, 1)],       # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]        # M7: B10 + B11
    ]

    # Strassen coefficients for W (C matrix)
    # List of lists, each inner list contains (i_loc, k_loc, coeff) for term s
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],       # M1: C00 + C11
        [(1, 0, 1), (1, 1, -1)],      # M2: C10 - C11
        [(0, 1, 1), (1, 1, 1)],       # M3: C01 + C11
        [(0, 0, 1), (1, 0, 1)],       # M4: C00 + C10
        [(0, 0, -1), (0, 1, 1)],      # M5: -C00 + C01
        [(1, 1, 1)],                  # M6: C11
        [(0, 0, 1)]                   # M7: C00
    ]

    col = 0
    # g: block index for j (0 or 1)
    # h: block index for k (0, 1, or 2)
    for g in range(2):
        for h in range(3):
            for s in range(7):
                # Update U
                for i_loc, j_loc, cf in U_coeffs[s]:
                    i = i_loc
                    j = g * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += cf

                # Update V
                for j_loc, k_loc, cf in V_coeffs[s]:
                    j = g * 2 + j_loc
                    k = h * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += cf

                # Update W
                for i_loc, k_loc, cf in W_coeffs[s]:
                    i = i_loc
                    k = h * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += cf

                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
