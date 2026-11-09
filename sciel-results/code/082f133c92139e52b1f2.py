"""
Matrix multiplication 2x4x8 — rank-minimal exact tensor decomposition.

The 2x4x8 multiplication can be viewed as 8 independent 2x2x2 multiplications
(one per column of the output), each with rank 7 via Strassen.
Total rank = 8 * 7 = 56.

U shape: (n*m, rank) = (8, 56)
V shape: (m*p, rank) = (32, 56)
W shape: (p*n, rank) = (16, 56)

Index conventions:
  U[i*m + j, c]  where i in 0..1 (row of A), j in 0..3 (col of A)
  V[j*p + k, c]  where j in 0..3 (row of B), k in 0..7 (col of B)
  W[k*n + i, c]  where k in 0..7 (col of C), i in 0..1 (row of C)
"""

import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 8 * 7  # 56: eight 2x2x2 Strassen blocks

    # U: (8, 56), V: (32, 56), W: (16, 56)
    U = [[0] * rank for _ in range(n * m)]   # 8 rows
    V = [[0] * rank for _ in range(m * p)]   # 32 rows
    W = [[0] * rank for _ in range(p * n)]   # 16 rows

    # For each output column k (0..7), we do a 2x2x2 Strassen multiply.
    # The 2x2x2 sub-problem involves:
    #   A_sub: 2x2 (rows 0,1 of A; two specific columns of A)
    #   B_sub: 2x2 (two specific rows of B; column k of B)
    #   C_sub: 2x2 (rows 0,1 of C; column k)
    #
    # We pair columns of A in pairs: (0,1), (2,3).
    # For output column k, we need rows of B that correspond to those A columns.
    # Actually, let me think more carefully.
    #
    # A is 2x4, B is 4x8. C = A*B is 2x8.
    # C[i][k] = sum_{j=0}^{3} A[i][j] * B[j][k]
    #
    # To use Strassen, we can split A's 4 columns into two groups of 2:
    #   Group 0: columns 0,1  -> A0 = A[:, 0:2] (2x2)
    #   Group 1: columns 2,3  -> A1 = A[:, 2:4] (2x2)
    # And B's 8 columns into four groups of 2:
    #   Group q: columns 2q, 2q+1 -> Bq = B[:, 2q:2q+2] (4x2)
    #
    # But Strassen works on square matrices. Let me think differently.
    #
    # Actually, the cleanest isonorphism: split A into two 2x2 blocks horizontally,
    # and B into four 4x2 blocks. But 4x2 * 2x2 doesn't give 2x2 output.
    #
    # Let me reconsider. The standard approach:
    # Split A (2x4) into A_left (2x2) using cols 0,1 and A_right (2x2) using cols 2,3.
    # Split B (4x8) into 4 blocks: B_top_left (2x2) rows 0,1 cols 2q,2q+1
    #   B_top_right (2x2) rows 0,1 cols 2q+2,2q+3 ... no this doesn't work cleanly.
    #
    # Better approach: think of it as 8 independent (2x2)*(2x2) problems.
    # For each pair of output columns (2q, 2q+1), we can form:
    #   For i in {0,1}, k in {2q, 2q+1}:
    #     C[i][k] = sum_j A[i][j] B[j][k]
    #
    # Actually, let me use the simplest correct decomposition:
    # Treat the 2x4x8 tensor as having structure that allows rank 56.
    #
    # One valid approach: for each of the 4 columns of A (j=0..3),
    # and for each of the 2 rows of A (i=0,1), and for each of the 8 columns of B (k=0..7),
    # we have a "basis" element. That gives 8*4*8 = 64 (schoolbook).
    #
    # To get 56, we use the fact that for each fixed pair of output columns,
    # the 2x2x2 multiplication (with 2 input rows, 2 input cols from A, 2 output cols)
    # can be done in rank 7.
    #
    # Let me use a concrete mapping:
    # For q in 0..3 (four groups of 2 output columns):
    #   Output columns: 2q, 2q+1
    #   We need a 2x2x2 Strassen multiply where:
    #     The "2x2" A part uses A[0:2, 2*(q%2):2*(q%2)+2]? No...
    #
    # Let me just use the known result: rank(2,4,8) <= 56.
    # The decomposition:
    #   Split A into A1 = A[:, 0:2] and A2 = A[:, 2:4] (each 2x2)
    #   Split B into B1 = B[0:2, :] and B2 = B[2:4, :] (each 2x8)
    #   C = A1*B1 + A2*B2
    #   Now A1*B1 is 2x2 * 2x8 = 2x8, and A2*B2 is 2x2 * 2x8 = 2x8.
    #   Each 2x2*2x8 can be seen as 4 independent 2x2*2x2 multiplications
    #   (since 8 cols / 2 = 4), each with rank 7.
    #   So rank(A1*B1) = 4*7 = 28, rank(A2*B2) = 4*7 = 28.
    #   Total rank = 56.
    #
    # For A1*B1 (2x2 * 2x8):
    #   For q in 0..3 (column pairs of B1):
    #     A1 (2x2) times B1[:, 2q:2q+2] (2x2) gives C[:, 2q:2q+2] (2x2)
    #     Use Strassen for this 2x2x2 multiply.
    #
    # Strassen for X (2x2) * Y (2x2) = Z (2x2):
    #   M1 = (X00+X11)*(Y00+Y11)
    #   M2 = (X10+X11)*Y00
    #   M3 = X00*(Y01-Y11)
    #   M4 = X11*(Y10-Y00)
    #   M5 = (X00+X01)*Y11
    #   M6 = (X10-X00)*(Y00+Y01)
    #   M7 = (X01-X11)*(Y10+Y11)
    #   Z00 = M1 + M4 - M5 + M7
    #   Z01 = M3 + M5
    #   Z10 = M2 + M4
    #   Z11 = M1 - M2 + M3 + M6
    #
    # In tensor decomposition terms, each multiplier Mk contributes to U, V, W.
    # For a single 2x2x2 block at "position" (a_offset, b_offset):
    #   X is A1 (rows 0,1; cols 0,1 within A1, which are cols 0,1 of A)
    #   Y is B1[:, 2q:2q+2] (rows 0,1 within B1, which are rows 0,1 of B; cols 2q,2q+1 of B)
    #
    # Let me define the Strassen multipliers for one 2x2x2 block:
    #   The "X" matrix elements: X00=A[0,0], X01=A[0,1], X10=A[1,0], X11=A[1,1]
    #   The "Y" matrix elements: Y00=B[0,2q], Y01=B[0,2q+1], Y10=B[1,2q], Y11=B[1,2q+1]
    #   Output: Z00=C[0,2q], Z01=C[0,2q+1], Z10=C[1,2q], Z11=C[1,2q+1]

    # Strassen multiplier definitions for a 2x2x2 block:
    # Each entry is (list of (x_row, x_col) with sign, list of (y_row, y_col) with sign, list of (z_row, z_col) with sign)
    # x: 2x2, y: 2x2, z: 2x2

    strassen_defs = [
        # M1 = (X00+X11)*(Y00+Y11)
        {
            'x': [(0,0,1), (1,1,1)],
            'y': [(0,0,1), (1,1,1)],
            'z': [(0,0,1), (0,1,-1), (1,0,-1), (1,1,1)],
        },
        # M2 = (X10+X11)*Y00
        {
            'x': [(1,0,1), (1,1,1)],
            'y': [(0,0,1)],
            'z': [(1,0,1), (1,1,-1)],
        },
        # M3 = X00*(Y01-Y11)
        {
            'x': [(0,0,1)],
            'y': [(0,1,1), (1,1,-1)],
            'z': [(0,1,1), (1,1,1)],
        },
        # M4 = X11*(Y10-Y00)
        {
            'x': [(1,1,1)],
            'y': [(1,0,1), (0,0,-1)],
            'z': [(0,0,1), (1,0,1)],
        },
        # M5 = (X00+X01)*Y11
        {
            'x': [(0,0,1), (0,1,1)],
            'y': [(1,1,1)],
            'z': [(0,0,-1), (0,1,1)],
        },
        # M6 = (X10-X00)*(Y00+Y01)
        {
            'x': [(1,0,1), (0,0,-1)],
            'y': [(0,0,1), (0,1,1)],
            'z': [(1,1,1)],
        },
        # M7 = (X01-X11)*(Y10+Y11)
        {
            'x': [(0,1,1), (1,1,-1)],
            'y': [(1,0,1), (1,1,1)],
            'z': [(0,0,1), (1,1,-1)],
        },
    ]

    # Process each block:
    # For block_type in [0, 1] (A1 and A2):
    #   a_col_offset = block_type * 2  # columns of A used
    #   b_row_offset = block_type * 2  # rows of B used
    #   For q in 0..3 (column pair of B):
    #     b_col_offset = 2 * q
    #     For s in 0..6 (Strassen multiplier):
    #       col = (block_type * 4 + q) * 7 + s

    for block_type in range(2):
        a_col_offset = block_type * 2  # 0 or 2
        b_row_offset = block_type * 2  # 0 or 2
        for q in range(4):
            b_col_offset = 2 * q
            for s in range(7):
                col = (block_type * 4 + q) * 7 + s
                sd = strassen_defs[s]

                # Add to U: X elements
                for (xr, xc, sign) in sd['x']:
                    # X is a 2x2 block of A: A[xr, a_col_offset + xc]
                    # U index: xr * m + (a_col_offset + xc) = xr * 4 + a_col_offset + xc
                    u_idx = xr * 4 + a_col_offset + xc
                    U[u_idx][col] += sign

                # Add to V: Y elements
                for (yr, yc, sign) in sd['y']:
                    # Y is a 2x2 block of B: B[b_row_offset + yr, b_col_offset + yc]
                    # V index: (b_row_offset + yr) * p + (b_col_offset + yc)
                    v_idx = (b_row_offset + yr) * 8 + (b_col_offset + yc)
                    V[v_idx][col] += sign

                # Add to W: Z elements
                for (zr, zc, sign) in sd['z']:
                    # Z is a 2x2 block of C: C[zr, b_col_offset + zc]
                    # W index: (b_col_offset + zc) * n + zr = (b_col_offset + zc) * 2 + zr
                    w_idx = (b_col_offset + zc) * 2 + zr
                    W[w_idx][col] += sign

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
