"""Rank-55 exact tensor decomposition for 2x3 * 3x10 matrix multiplication
using block-wise Strassen (rank 7 per 2x2x2 block) + naive (rank 4 per 2x1x2 block)."""

DIMENSIONS = [2, 3, 10]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # Split the 10 output columns into 5 blocks of 2
    num_blocks = p // 2  # 5
    # Each block: 7 (Strassen) + 4 (naive) = 11 terms
    # Total rank = 5 * 11 = 55
    rank = num_blocks * 11  # 55

    U = np.zeros((n * m, rank))  # (6, 55)
    V = np.zeros((m * p, rank))  # (30, 55)
    W = np.zeros((p * n, rank))  # (20, 55)

    # Strassen formulas for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    #
    # Naive for 2x1x2:
    # N1 = A13 * B31
    # N2 = A13 * B32
    # N3 = A23 * B31
    # N4 = A23 * B32
    #
    # C11 += N1, C12 += N2, C21 += N3, C22 += N4

    # Indexing: A[i,j] with i in 0..1, j in 0..2
    # B[j,k] with j in 0..2, k in 0..9
    # C[i,k] with i in 0..1, k in 0..9
    #
    # U row index: i*m + j = i*3 + j
    # V row index: j*p + k = j*10 + k
    # W row index: k*n + i = k*2 + i

    # Strassen terms: (a_coeff, b_coeff, c_row, c_col, sign)
    # where a_coeff is (i,j,coeff), b_coeff is (j,k,coeff)
    # For each block b, columns k1=2b, k2=2b+1

    # Strassen M1: (A00+A11)(B0k1+B0k2 + B1k1+B1k2) -> wait, need to be careful
    # Actually for 2x2x2: A is 2x2, B is 2x2, C is 2x2
    # A[i][j] where i=0,1 and j=0,1
    # B[j][k] where j=0,1 and k=0,1
    # C[i][k] where i=0,1 and k=0,1

    # In our context for block b:
    # A submatrix: rows 0,1; cols 0,1 (i.e., A[0,0], A[0,1], A[1,0], A[1,1])
    # B submatrix: rows 0,1; cols 2b, 2b+1 (i.e., B[0,2b], B[0,2b+1], B[1,2b], B[1,2b+1])
    # C submatrix: rows 0,1; cols 2b, 2b+1 (i.e., C[0,2b], C[0,2b+1], C[1,2b], C[1,2b+1])

    # Strassen terms for 2x2x2:
    # M1 = (A00+A11)(B00+B01+B10+B11) -- no wait
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6

    # Let me define the Strassen terms as tuples:
    # (u_factors, v_factors, w_row, w_col, coeff)
    # u_factors: list of (row_in_A, coeff) where row_in_A is (i,j)
    # v_factors: list of (col_in_B, coeff) where col_in_B is (j,k)
    # w_row: i, w_col: k (local indices 0,1)
    # coeff: the coefficient in the sum

    # For block b, local k=0 maps to global k=2b, local k=1 maps to global k=2b+1

    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        {
            'u': [((0,0), 1), ((1,1), 1)],
            'v': [((0,0), 1), ((1,1), 1)],
            'w': [(0,0, 1), (1,1, 1)],  # C00 += M1, C11 += M1
        },
        # M2 = (A10+A11)(B00)
        {
            'u': [((1,0), 1), ((1,1), 1)],
            'v': [((0,0), 1)],
            'w': [(1,0, 1)],  # C10 += M2
        },
        # M3 = A00(B01-B11)
        {
            'u': [((0,0), 1)],
            'v': [((0,1), 1), ((1,1), -1)],
            'w': [(0,1, 1)],  # C01 += M3
        },
        # M4 = A11(B10-B00)
        {
            'u': [((1,1), 1)],
            'v': [((1,0), 1), ((0,0), -1)],
            'w': [(0,0, -1), (1,0, 1)],  # C00 -= M4, C10 += M4
        },
        # M5 = (A00+A01)(B11)
        {
            'u': [((0,0), 1), ((0,1), 1)],
            'v': [((1,1), 1)],
            'w': [(0,0, 1)],  # C00 += M5
        },
        # M6 = (A10-A00)(B00+B01)
        {
            'u': [((1,0), 1), ((0,0), -1)],
            'v': [((0,0), 1), ((0,1), 1)],
            'w': [(1,1, -1)],  # C11 -= M6
        },
        # M7 = (A01-A11)(B10+B11)
        {
            'u': [((0,1), 1), ((1,1), -1)],
            'v': [((1,0), 1), ((1,1), 1)],
            'w': [(0,0, -1), (0,1, 1)],  # C00 -= M7, C01 += M7
        },
    ]

    # Naive terms for 2x1x2:
    # N1 = A02 * B20 -> C00
    # N2 = A02 * B21 -> C01
    # N3 = A12 * B20 -> C10
    # N4 = A12 * B21 -> C11

    naive_terms = [
        {
            'u': [((0,2), 1)],
            'v': [((2,0), 1)],
            'w': [(0,0, 1)],
        },
        {
            'u': [((0,2), 1)],
            'v': [((2,1), 1)],
            'w': [(0,1, 1)],
        },
        {
            'u': [((1,2), 1)],
            'v': [((2,0), 1)],
            'w': [(1,0, 1)],
        },
        {
            'u': [((1,2), 1)],
            'v': [((2,1), 1)],
            'w': [(1,1, 1)],
        },
    ]

    col_offset = 0
    for b in range(num_blocks):
        k1 = 2 * b
        k2 = 2 * b + 1

        # Process Strassen terms
        for term in strassen_terms:
            col = col_offset
            # U contributions
            for (ai, aj), coeff in term['u']:
                u_row = ai * m + aj
                U[u_row, col] += coeff

            # V contributions
            for (bj, bk_local), coeff in term['v']:
                # bk_local is 0 or 1, maps to k1 or k2
                bk_global = k1 if bk_local == 0 else k2
                v_row = bj * p + bk_global
                V[v_row, col] += coeff

            # W contributions
            for (ci, ck_local), coeff in term['w']:
                ck_global = k1 if ck_local == 0 else k2
                w_row = ck_global * n + ci
                W[w_row, col] += coeff

            col_offset += 1

        # Process naive terms
        for term in naive_terms:
            col = col_offset
            # U contributions
            for (ai, aj), coeff in term['u']:
                u_row = ai * m + aj
                U[u_row, col] += coeff

            # V contributions
            for (bj, bk_local), coeff in term['v']:
                bk_global = k1 if bk_local == 0 else k2
                v_row = bj * p + bk_global
                V[v_row, col] += coeff

            # W contributions
            for (ci, ck_local), coeff in term['w']:
                ck_global = k1 if ck_local == 0 else k2
                w_row = ck_global * n + ci
                W[w_row, col] += coeff

            col_offset += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
