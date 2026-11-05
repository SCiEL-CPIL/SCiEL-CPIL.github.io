"""Rewritten program for the 2x4x7 matrix multiplication tensor decomposition task.

Uses a block decomposition:
- 6 Strassen 2x2x2 blocks (rank 7 each) for the 2x4x6 portion = 42
- 1 combined dot-product block for the 2x4x1 portion = 8
Total rank: 50
"""
import numpy as np

DIMENSIONS = [2, 4, 7]


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * rank for _ in range(p * n)]  # 14 x 50

    col_offset = 0

    def add_strassen(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """Add a Strassen 2x2x2 block.
        A is 2x2 at rows [a_row_start, a_row_start+1], cols [a_col_start, a_col_start+1]
        B is 2x2 at rows [b_row_start, b_row_start+1], cols [b_col_start, b_col_start+1]
        C is 2x2 at rows [c_row_start, c_row_start+1], cols [c_col_start, c_col_start+1]
        
        U index: (i*m + j) for A[i][j]
        V index: (k*p + l) for B[k][l]
        W index: (l*n + i) for C[i][l]  (trace convention: W[k*n+i, r] where k=col, i=row)
        """
        nonlocal col_offset

        # M1: (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
        U[a_row_start * m + a_col_start][col_offset] += 1
        U[(a_row_start + 1) * m + (a_col_start + 1)][col_offset] += 1
        V[b_row_start * p + b_col_start][col_offset] += 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col_offset] += 1
        W[c_col_start * n + c_row_start][col_offset] += 1
        W[(c_col_start + 1) * n + (c_row_start + 1)][col_offset] += 1
        col_offset += 1

        # M2: (A10 + A11) * B00 -> C10 += M2, C11 -= M2
        U[(a_row_start + 1) * m + a_col_start][col_offset] += 1
        U[(a_row_start + 1) * m + (a_col_start + 1)][col_offset] += 1
        V[b_row_start * p + b_col_start][col_offset] += 1
        W[c_col_start * n + (c_row_start + 1)][col_offset] += 1
        W[(c_col_start + 1) * n + (c_row_start + 1)][col_offset] -= 1
        col_offset += 1

        # M3: A00 * (B01 - B11) -> C01 += M3, C11 += M3
        U[a_row_start * m + a_col_start][col_offset] += 1
        V[b_row_start * p + (b_col_start + 1)][col_offset] += 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col_offset] -= 1
        W[(c_col_start + 1) * n + c_row_start][col_offset] += 1
        W[(c_col_start + 1) * n + (c_row_start + 1)][col_offset] += 1
        col_offset += 1

        # M4: A11 * (B10 - B00) -> C00 += M4, C10 += M4
        U[(a_row_start + 1) * m + (a_col_start + 1)][col_offset] += 1
        V[(b_row_start + 1) * p + b_col_start][col_offset] += 1
        V[b_row_start * p + b_col_start][col_offset] -= 1
        W[c_col_start * n + c_row_start][col_offset] += 1
        W[c_col_start * n + (c_row_start + 1)][col_offset] += 1
        col_offset += 1

        # M5: (A00 + A01) * B11 -> C00 -= M5, C01 += M5
        U[a_row_start * m + a_col_start][col_offset] += 1
        U[a_row_start * m + (a_col_start + 1)][col_offset] += 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col_offset] += 1
        W[c_col_start * n + c_row_start][col_offset] -= 1
        W[(c_col_start + 1) * n + c_row_start][col_offset] += 1
        col_offset += 1

        # M6: (A10 - A00) * (B00 + B01) -> C11 += M6
        U[(a_row_start + 1) * m + a_col_start][col_offset] += 1
        U[a_row_start * m + a_col_start][col_offset] -= 1
        V[b_row_start * p + b_col_start][col_offset] += 1
        V[b_row_start * p + (b_col_start + 1)][col_offset] += 1
        W[(c_col_start + 1) * n + (c_row_start + 1)][col_offset] += 1
        col_offset += 1

        # M7: (A01 - A11) * (B10 + B11) -> C00 += M7
        U[a_row_start * m + (a_col_start + 1)][col_offset] += 1
        U[(a_row_start + 1) * m + (a_col_start + 1)][col_offset] -= 1
        V[(b_row_start + 1) * p + b_col_start][col_offset] += 1
        V[(b_row_start + 1) * p + (b_col_start + 1)][col_offset] += 1
        W[c_col_start * n + c_row_start][col_offset] += 1
        col_offset += 1

    def add_dot(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        """Add a 2x2x1 dot product block (4 terms).
        A is 2x2, B is 2x1, C is 2x1.
        C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        """
        nonlocal col_offset

        # A00 * B00 -> C0
        U[a_row_start * m + a_col_start][col_offset] += 1
        V[b_row_start * p + b_col_start][col_offset] += 1
        W[c_col_start * n + c_row_start][col_offset] += 1
        col_offset += 1

        # A01 * B10 -> C0
        U[a_row_start * m + (a_col_start + 1)][col_offset] += 1
        V[(b_row_start + 1) * p + b_col_start][col_offset] += 1
        W[c_col_start * n + c_row_start][col_offset] += 1
        col_offset += 1

        # A10 * B00 -> C1
        U[(a_row_start + 1) * m + a_col_start][col_offset] += 1
        V[b_row_start * p + b_col_start][col_offset] += 1
        W[c_col_start * n + (c_row_start + 1)][col_offset] += 1
        col_offset += 1

        # A11 * B10 -> C1
        U[(a_row_start + 1) * m + (a_col_start + 1)][col_offset] += 1
        V[(b_row_start + 1) * p + b_col_start][col_offset] += 1
        W[c_col_start * n + (c_row_start + 1)][col_offset] += 1
        col_offset += 1

    # The 2x4x7 matrix multiplication can be decomposed as:
    # A (2x4) * B (4x7) = C (2x7)
    #
    # Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    # Split B into B1 (4x2, cols 0-1), B2 (4x2, cols 2-3), B3 (4x2, cols 4-5), B4 (4x1, col 6)
    #
    # C = A1*B1 + A1*B2 + A1*B3 + A2*B1 + A2*B2 + A2*B3 + A1*B4 + A2*B4
    #
    # Using Strassen for the 2x2x2 blocks (6 blocks * 7 = 42)
    # Using dot product for the 2x2x1 blocks (2 blocks * 4 = 8)
    # Total: 50

    # A1*B1: A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    add_strassen(0, 0, 0, 0, 0, 0)

    # A1*B2: A rows 0-1, cols 0-1; B rows 0-1, cols 2-3; C rows 0-1, cols 2-3
    add_strassen(0, 0, 0, 2, 0, 2)

    # A1*B3: A rows 0-1, cols 0-1; B rows 0-1, cols 4-5; C rows 0-1, cols 4-5
    add_strassen(0, 0, 0, 4, 0, 4)

    # A2*B1: A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C rows 0-1, cols 0-1
    add_strassen(0, 2, 2, 0, 0, 0)

    # A2*B2: A rows 0-1, cols 2-3; B rows 2-3, cols 2-3; C rows 0-1, cols 2-3
    add_strassen(0, 2, 2, 2, 0, 2)

    # A2*B3: A rows 0-1, cols 2-3; B rows 2-3, cols 4-5; C rows 0-1, cols 4-5
    add_strassen(0, 2, 2, 4, 0, 4)

    # A1*B4: A rows 0-1, cols 0-1; B rows 0-1, col 6; C rows 0-1, col 6
    add_dot(0, 0, 0, 6, 0, 6)

    # A2*B4: A rows 0-1, cols 2-3; B rows 2-3, col 6; C rows 0-1, col 6
    add_dot(0, 2, 2, 6, 0, 6)

    return {"U": U, "V": V, "W": W}
