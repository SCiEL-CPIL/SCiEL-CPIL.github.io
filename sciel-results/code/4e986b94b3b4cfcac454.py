"""Matrix multiplication 2,2,6 - rank-21 Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-21 Strassen decomposition for M_{2,2,6}.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    return _build_rank21(n, m, p)


def _build_rank21(n, m, p):
    """Build the rank-21 Strassen decomposition for 2x2 * 2x6."""
    rank = 21

    # U shape: (n*m, rank) = (4, 21)
    # V shape: (m*p, rank) = (12, 21)
    # W shape: (p*n, rank) = (12, 21)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 7 multiplications for a 2x2 block:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b10)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 - M4 + M7
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6

    # Local indices for U (A is 2x2, flattened row-major: a00=0, a01=1, a10=2, a11=3)
    # Local indices for V (B block is 2x2, flattened: b00=0, b01=1, b10=2, b11=3)
    # Local indices for W (C block is 2x2, flattened: c00=0, c01=1, c10=2, c11=3)

    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1 = (a00+a11)(b00+b11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (a10+a11)b00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = a00(b10-b11)
        ({0: 1}, {2: 1, 3: -1}, {2: 1}),
        # M4 = a11(b01-b10)
        ({3: 1}, {1: 1, 2: -1}, {0: -1, 1: 1}),
        # M5 = (a00+a01)b11
        ({0: 1, 1: 1}, {3: 1}, {1: 1}),
        # M6 = (a10-a00)(b00+b01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (a01-a11)(b10+b11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # For each of the 3 blocks (b=0,1,2) corresponding to columns 2b, 2b+1 of B
    # and output columns 2b, 2b+1 of C:
    for b in range(3):
        for t in range(7):
            r = b * 7 + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]

            # Fill U
            for idx, val in u_coeffs.items():
                U[idx][r] = val

            # Fill V: local index l maps to global index (row in m*p)
            # B is 2x6, so m*p = 12. Row i of B, column j.
            # Local b00 -> row 0, col 2b -> index 0*6 + 2b
            # Local b01 -> row 0, col 2b+1 -> index 0*6 + 2b+1
            # Local b10 -> row 1, col 2b -> index 1*6 + 2b
            # Local b11 -> row 1, col 2b+1 -> index 1*6 + 2b+1
            v_map = {0: 0 * p + 2 * b, 1: 0 * p + 2 * b + 1,
                     2: 1 * p + 2 * b, 3: 1 * p + 2 * b + 1}
            for l, val in v_coeffs.items():
                V[v_map[l]][r] = val

            # Fill W: local index for C block
            # C is 2x6, p*n = 12. Convention: W[k*n+i, r] where k is col, i is row.
            # Local c00 -> row 0, col 2b -> k=2b, i=0 -> index (2b)*n + 0
            # Local c01 -> row 0, col 2b+1 -> k=2b+1, i=0 -> index (2b+1)*n + 0
            # Local c10 -> row 1, col 2b -> k=2b, i=1 -> index (2b)*n + 1
            # Local c11 -> row 1, col 2b+1 -> k=2b+1, i=1 -> index (2b+1)*n + 1
            w_map = {0: (2 * b) * n + 0, 1: (2 * b + 1) * n + 0,
                     2: (2 * b) * n + 1, 3: (2 * b + 1) * n + 1}
            for idx, val in w_coeffs.items():
                W[w_map[idx]][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
