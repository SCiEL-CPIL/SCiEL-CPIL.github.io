"""Rewritten: direct, explicit rank-42 decomposition for 2x4 x 4x6 matrix multiplication.

Strategy: partition B into three 4x2 column-blocks. Each block requires a 2x4 x 4x2
multiplication (16 outputs). We use a 14-rank decomposition per block (two overlapping
2x2x2 Strassen instances sharing 2 products), giving total rank 3*14 = 42.
"""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42

    # U: (n*m=8) x rank, V: (m*p=24) x rank, W: (p*n=12) x rank
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # For each 2x4 x 4x2 sub-problem (A 2x4, B 4x2, C 2x2):
    # We split A into two 2x2 blocks (left A_L cols 0-1, right A_R cols 2-3)
    # We split B into two 2x2 blocks (top B_T rows 0-1, bottom B_B rows 2-3)
    # C = A_L @ B_T + A_R @ B_B
    #
    # Using two Strassen instances (7 products each = 14), but we can share
    # 2 products to get 14? Actually Strassen is 7 per 2x2x2, so two = 14.
    # No sharing needed; just 14 products per block.
    #
    # Strassen 7 products for multiplying 2x2 A by 2x2 B to get 2x2 C:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # Indexing within 2x2: (row, col) -> idx = row*2 + col
    #   00->0, 01->1, 10->2, 11->3

    # Define the 7 Strassen products as (U_coeffs, V_coeffs, W_coeffs)
    # Each is a dict mapping local 2x2 index to coefficient
    STRASSEN = [
        # P1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # P2: (A10+A11)B00 -> C10 += 1, C11 -= 1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # P3: A00(B01-B11) -> C01 += 1, C11 += 1
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # P4: A11(B10-B00) -> C00 += 1, C10 += 1
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # P5: (A00+A01)B11 -> C00 -= 1, C01 += 1
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # P6: (A10-A00)(B00+B01) -> C11 += 1
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # P7: (A01-A11)(B10+B11) -> C00 += 1
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    col = 0
    for block in range(3):
        # B column offset for this block
        b_col = block * 2
        # B row offsets for top/bottom halves
        b_top_row = 0
        b_bot_row = 2

        # A column offsets for left/right halves
        a_left_col = 0
        a_right_col = 2

        # --- Strassen instance 1: A_left (2x2) * B_top (2x2) -> C contribution ---
        for u_c, v_c, w_c in STRASSEN:
            for idx, coeff in u_c.items():
                r, c = divmod(idx, 2)
                u_row = r * m + (a_left_col + c)
                U[u_row][col] += coeff
            for idx, coeff in v_c.items():
                r, c = divmod(idx, 2)
                v_row = (b_top_row + r) * p + (b_col + c)
                V[v_row][col] += coeff
            for idx, coeff in w_c.items():
                r, c = divmod(idx, 2)
                # W[k*n + i] where k=output col, i=output row
                w_row = (b_col + c) * n + r
                W[w_row][col] += coeff
            col += 1

        # --- Strassen instance 2: A_right (2x2) * B_bottom (2x2) -> C contribution ---
        for u_c, v_c, w_c in STRASSEN:
            for idx, coeff in u_c.items():
                r, c = divmod(idx, 2)
                u_row = r * m + (a_right_col + c)
                U[u_row][col] += coeff
            for idx, coeff in v_c.items():
                r, c = divmod(idx, 2)
                v_row = (b_bot_row + r) * p + (b_col + c)
                V[v_row][col] += coeff
            for idx, coeff in w_c.items():
                r, c = divmod(idx, 2)
                w_row = (b_col + c) * n + r
                W[w_row][col] += coeff
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
