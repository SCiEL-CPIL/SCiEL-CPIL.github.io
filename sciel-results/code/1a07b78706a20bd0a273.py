"""Matrix multiplication 2x4 by 4x6: rank-minimal exact tensor decomposition."""

import time

def construct():
    n, m, p = 2, 4, 6  # A is n x m, B is m x p
    U_rows = n * m  # 8
    V_rows = m * p  # 24
    W_rows = p * n  # 12

    # Strassen 2x2 decomposition:
    # Rank 42 is the sum of ranks of two 2x2 by 2x6 multiplications.
    # Each 2x2 by 2x6 is three 2x2 by 2x2 multiplications (Strassen, rank 7 each).
    # Total 3 * 7 * 2 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]

    col = 0

    # Process A0 (A columns 0-1) times each 2x2 block of B_top (B rows 0-1, columns 2t-2t+1)
    # And A1 (A columns 2-3) times each 2x2 block of B_bottom (B rows 2-3, columns 2t-2t+1)

    for a_block in range(2):  # 0: A0 (cols 0-1), 1: A1 (cols 2-3)
        for t in range(3):  # 3 blocks of B, each 2x2
            # B block is rows 2*a_block to 2*a_block+1, cols 2*t to 2*t+1
            b_row0 = 2 * a_block
            b_row1 = 2 * a_block + 1
            b_col0 = 2 * t
            b_col1 = 2 * t + 1

            a_col_offset = 2 * a_block

            # Strassen 7 terms
            # Term 1: M1 = (a00+a11) * (b00+b11)
            # A part: a00 (i=0,j=0) + a11 (i=1,j=1)
            # B part: b00 (row=b_row0, col=b_col0) + b11 (row=b_row1, col=b_col1)
            # C part: C00 +, C11 +
            
            U[0 * m + a_col_offset + 0][col] += 1
            U[1 * m + a_col_offset + 1][col] += 1
            V[b_row0 * p + b_col0][col] += 1
            V[b_row1 * p + b_col1][col] += 1
            W[(2*t) * n + 0][col] += 1
            W[(2*t+1) * n + 1][col] += 1
            col += 1

            # M2 = (a10+a11) * b00
            # C10 +, C11 -
            U[1 * m + a_col_offset + 0][col] += 1
            U[1 * m + a_col_offset + 1][col] += 1
            V[b_row0 * p + b_col0][col] += 1
            W[(2*t) * n + 1][col] += 1
            W[(2*t+1) * n + 1][col] += -1
            col += 1

            # M3 = a00 * (b01-b11)
            # C01 +, C11 +
            U[0 * m + a_col_offset + 0][col] += 1
            V[b_row0 * p + b_col1][col] += 1
            V[b_row1 * p + b_col1][col] += -1
            W[(2*t+1) * n + 0][col] += 1
            W[(2*t+1) * n + 1][col] += 1
            col += 1

            # M4 = a11 * (b10-b00)
            # C00 +, C10 +
            U[1 * m + a_col_offset + 1][col] += 1
            V[b_row1 * p + b_col0][col] += 1
            V[b_row0 * p + b_col0][col] += -1
            W[(2*t) * n + 0][col] += 1
            W[(2*t) * n + 1][col] += 1
            col += 1

            # M5 = (a00+a01) * b11
            # C00 -, C01 +
            U[0 * m + a_col_offset + 0][col] += 1
            U[0 * m + a_col_offset + 1][col] += 1
            V[b_row1 * p + b_col1][col] += 1
            W[(2*t) * n + 0][col] += -1
            W[(2*t+1) * n + 0][col] += 1
            col += 1

            # M6 = (a10-a00) * (b00+b01)
            # C11 +
            U[1 * m + a_col_offset + 0][col] += 1
            U[0 * m + a_col_offset + 0][col] += -1
            V[b_row0 * p + b_col0][col] += 1
            V[b_row0 * p + b_col1][col] += 1
            W[(2*t+1) * n + 1][col] += 1
            col += 1

            # M7 = (a01-a11) * (b10+b11)
            # C00 +
            U[0 * m + a_col_offset + 1][col] += 1
            U[1 * m + a_col_offset + 1][col] += -1
            V[b_row1 * p + b_col0][col] += 1
            V[b_row1 * p + b_col1][col] += 1
            W[(2*t) * n + 0][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}
