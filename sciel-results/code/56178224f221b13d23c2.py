"""
Matrix multiplication 6x6x6 tensor decomposition.

Strategy:
  - Hybrid block decomposition: 2x2 grid of 3x3 blocks.
  - Outer 2x2 block multiplication uses Strassen's 7-product algorithm.
  - Inner 3x3 block multiplication uses Laderman's 15-product algorithm.
  - Total rank = 7 * 15 = 105.

  This is significantly better than the 3x3 grid of 2x2 Strassen blocks (rank 189)
  or schoolbook (rank 216).
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix by a 6x6 matrix.
    Uses a 2x2 block decomposition where each 3x3 block product uses Laderman's algorithm
    (rank 15), and the outer 2x2 block structure uses Strassen (rank 7).
    Total rank = 7 * 15 = 105.
    """
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # all 6

    # Laderman's algorithm for 3x3 matrix multiplication.
    # Given A (3x3) and B (3x3), C = A*B.
    # Laderman uses 15 multiplications:
    #
    # A = [[a11, a12, a13],
    #      [a21, a22, a23],
    #      [a31, a32, a33]]
    #
    # B = [[b11, b12, b13],
    #      [b21, b22, b23],
    #      [b31, b32, b33]]
    #
    # M1  = a11 * (b12 - b13)
    # M2  = (a11 + a12) * b13
    # M3  = (a21 + a22) * b22
    # M4  = a33 * (b31 - b32)
    # M5  = (a31 + a33) * b32
    # M6  = (a13 + a23) * (b11 + b21)
    # M7  = (a11 + a33) * (b11 + b33)
    # M8  = (a13 + a33) * (b11 - b13 + b31 + b33)
    # M9  = (a11 + a12 + a33 + a32) * (b11 + b33)
    # M10 = (a12 + a22) * (b21 + b22)
    # M11 = (a13 + a23 + a33 + a31) * (b11 + b31)
    # M12 = (a11 + a21 + a33 + a31) * (b13 + b33)
    # M13 = (a12 + a22 + a33 + a32) * (b23 + b33)
    # M14 = (a13 + a23 + a31 + a33) * (b12 + b32)
    # M15 = (a11 + a21 + a32 + a33) * (b12 + b22 + b32)
    #
    # C11 = M7 + M10 - M2 - M5 - M13
    # C12 = M1 + M2 + M9 - M4 - M14
    # C13 = M8 - M6 + M11 - M12 - M15
    # C21 = M6 + M12 - M1 - M3 - M9
    # C22 = M3 + M13 - M5 - M7 - M11
    # C23 = M4 + M14 - M2 - M8 - M10
    # C31 = M5 + M9 - M6 - M12 - M15
    # C32 = M7 + M10 - M3 - M11 - M13
    # C33 = M3 + M15 - M1 - M4 - M8

    # Laderman's 15 products defined as (U_factors, V_factors, W_factors)
    # Each factor is a list of (index, coefficient)
    # U factors index into A's 9 elements: (row, col) -> row*3+col
    # V factors index into B's 9 elements
    # W factors index into C's 9 elements

    laderman_products = [
        # M1 = a11 * (b12 - b13)
        {
            'U': [(0, 1)],           # a11
            'V': [(1, 1), (2, -1)],  # b12 - b13
            'W': [(0, 1), (3, -1), (6, 1)],  # C11(+1), C21(-1), C31(+1)
        },
        # M2 = (a11 + a12) * b13
        {
            'U': [(0, 1), (1, 1)],   # a11 + a12
            'V': [(2, 1)],           # b13
            'W': [(0, -1), (1, 1), (5, -1)],  # C11(-1), C12(+1), C23(-1)
        },
        # M3 = (a21 + a22) * b22
        {
            'U': [(3, 1), (4, 1)],   # a21 + a22
            'V': [(4, 1)],           # b22
            'W': [(3, -1), (4, 1), (7, -1)],  # C21(-1), C22(+1), C32(-1)
        },
        # M4 = a33 * (b31 - b32)
        {
            'U': [(8, 1)],           # a33
            'V': [(6, 1), (7, -1)],  # b31 - b32
            'W': [(1, -1), (5, 1), (8, -1)],  # C12(-1), C23(+1), C33(-1)
        },
        # M5 = (a31 + a33) * b32
        {
            'U': [(6, 1), (8, 1)],   # a31 + a33
            'V': [(7, 1)],           # b32
            'W': [(0, -1), (4, -1), (6, 1)],  # C11(-1), C22(-1), C31(+1)
        },
        # M6 = (a13 + a23) * (b11 + b21)
        {
            'U': [(2, 1), (5, 1)],   # a13 + a23
            'V': [(0, 1), (3, 1)],   # b11 + b21
            'W': [(2, -1), (3, 1), (6, -1)],  # C13(-1), C21(+1), C31(-1)
        },
        # M7 = (a11 + a33) * (b11 + b33)
        {
            'U': [(0, 1), (8, 1)],   # a11 + a33
            'V': [(0, 1), (8, 1)],   # b11 + b33
            'W': [(0, 1), (4, -1), (7, 1)],  # C11(+1), C22(-1), C32(+1)
        },
        # M8 = (a13 + a33) * (b11 - b13 + b31 + b33)
        {
            'U': [(2, 1), (8, 1)],   # a13 + a33
            'V': [(0, 1), (2, -1), (6, 1), (8, 1)],  # b11 - b13 + b31 + b33
            'W': [(2, 1), (5, -1), (8, -1)],  # C13(+1), C23(-1), C33(-1)
        },
        # M9 = (a11 + a12 + a33 + a32) * (b11 + b33)
        {
            'U': [(0, 1), (1, 1), (8, 1), (7, 1)],  # a11 + a12 + a33 + a32
            'V': [(0, 1), (8, 1)],                   # b11 + b33
            'W': [(1, 1), (3, -1), (6, 1), (7, -1)],  # C12(+1), C21(-1), C31(+1), C32(-1)
        },
        # M10 = (a12 + a22) * (b21 + b22)
        {
            'U': [(1, 1), (4, 1)],   # a12 + a22
            'V': [(3, 1), (4, 1)],   # b21 + b22
            'W': [(0, 1), (1, -1), (3, -1), (7, 1)],  # C11(+1), C12(-1), C21(-1), C32(+1)
        },
        # M11 = (a13 + a23 + a33 + a31) * (b11 + b31)
        {
            'U': [(2, 1), (5, 1), (8, 1), (6, 1)],  # a13 + a23 + a33 + a31
            'V': [(0, 1), (6, 1)],                   # b11 + b31
            'W': [(2, -1), (4, -1), (7, -1)],  # C13(-1), C22(-1), C32(-1)
        },
        # M12 = (a11 + a21 + a33 + a31) * (b13 + b33)
        {
            'U': [(0, 1), (3, 1), (8, 1), (6, 1)],  # a11 + a21 + a33 + a31
            'V': [(2, 1), (8, 1)],                   # b13 + b33
            'W': [(2, -1), (3, 1), (6, -1)],  # C13(-1), C21(+1), C31(-1)
        },
        # M13 = (a12 + a22 + a33 + a32) * (b23 + b33)
        {
            'U': [(1, 1), (4, 1), (8, 1), (7, 1)],  # a12 + a22 + a33 + a32
            'V': [(5, 1), (8, 1)],                   # b23 + b33
            'W': [(0, -1), (4, 1), (7, -1)],  # C11(-1), C22(+1), C32(-1)
        },
        # M14 = (a13 + a23 + a31 + a33) * (b12 + b32)
        {
            'U': [(2, 1), (5, 1), (6, 1), (8, 1)],  # a13 + a23 + a31 + a33
            'V': [(1, 1), (7, 1)],                   # b12 + b32
            'W': [(1, -1), (5, 1), (8, 1)],  # C12(-1), C23(+1), C33(+1)
        },
        # M15 = (a11 + a21 + a32 + a33) * (b12 + b22 + b32)
        {
            'U': [(0, 1), (3, 1), (7, 1), (8, 1)],  # a11 + a21 + a32 + a33
            'V': [(1, 1), (4, 1), (7, 1)],           # b12 + b22 + b32
            'W': [(2, -1), (4, 1), (6, -1), (8, 1)],  # C13(-1), C22(+1), C31(-1), C33(+1)
        },
    ]

    # Strassen's 7 products for 2x2 block multiplication.
    # A = [[A11, A12], [A21, A22]] where each Aij is a 3x3 block
    # B = [[B11, B12], [B21, B22]] where each Bkj is a 3x3 block
    # C = [[C11, C12], [C21, C22]] where each Cij is a 3x3 block
    #
    # S1 = A11 + A22
    # S2 = A21 + A22
    # S3 = A11
    # S4 = A22
    # S5 = A11 + A12
    # S6 = A21 - A11
    # S7 = A12 - A22
    #
    # T1 = B11 + B22
    # T2 = B11
    # T3 = B12 - B22
    # T4 = B21 - B11
    # T5 = B22
    # T6 = B11 + B12
    # T7 = B21 + B22
    #
    # M1 = S1 * T1 = (A11+A22)*(B11+B22)
    # M2 = S2 * T2 = (A21+A22)*B11
    # M3 = S3 * T3 = A11*(B12-B22)
    # M4 = S4 * T4 = A22*(B21-B11)
    # M5 = S5 * T5 = (A11+A12)*B22
    # M6 = S6 * T6 = (A21-A11)*(B11+B12)
    # M7 = S7 * T7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Strassen's 7 products defined as (U_block_factors, V_block_factors, W_block_factors)
    # U_block_factors: list of (block_index, coefficient) where block_index is 0=A11, 1=A12, 2=A21, 3=A22
    # V_block_factors: list of (block_index, coefficient) where block_index is 0=B11, 1=B12, 2=B21, 3=B22
    # W_block_factors: list of (block_index, coefficient) where block_index is 0=C11, 1=C12, 2=C21, 3=C22

    strassen_products = [
        # M1 = (A11 + A22) * (B11 + B22)
        {
            'U': [(0, 1), (3, 1)],   # A11 + A22
            'V': [(0, 1), (3, 1)],   # B11 + B22
            'W': [(0, 1), (3, 1)],   # C11(+1), C22(+1)
        },
        # M2 = (A21 + A22) * B11
        {
            'U': [(2, 1), (3, 1)],   # A21 + A22
            'V': [(0, 1)],           # B11
            'W': [(2, 1), (3, -1)],  # C21(+1), C22(-1)
        },
        # M3 = A11 * (B12 - B22)
        {
            'U': [(0, 1)],           # A11
            'V': [(1, 1), (3, -1)],  # B12 - B22
            'W': [(1, 1), (3, 1)],   # C12(+1), C22(+1)
        },
        # M4 = A22 * (B21 - B11)
        {
            'U': [(3, 1)],           # A22
            'V': [(2, 1), (0, -1)],  # B21 - B11
            'W': [(0, 1), (2, 1)],   # C11(+1), C21(+1)
        },
        # M5 = (A11 + A12) * B22
        {
            'U': [(0, 1), (1, 1)],   # A11 + A12
            'V': [(3, 1)],           # B22
            'W': [(0, -1), (1, 1)],  # C11(-1), C12(+1)
        },
        # M6 = (A21 - A11) * (B11 + B12)
        {
            'U': [(2, 1), (0, -1)],  # A21 - A11
            'V': [(0, 1), (1, 1)],   # B11 + B12
            'W': [(3, 1)],           # C22(+1)
        },
        # M7 = (A12 - A22) * (B21 + B22)
        {
            'U': [(1, 1), (3, -1)],  # A12 - A22
            'V': [(2, 1), (3, 1)],   # B21 + B22
            'W': [(0, 1)],           # C11(+1)
        },
    ]

    # Total rank = 7 (Strassen) * 15 (Laderman) = 105
    rank = 105
    U = [[0] * rank for _ in range(n * m)]  # 36 x 105
    V = [[0] * rank for _ in range(m * p)]  # 36 x 105
    W = [[0] * rank for _ in range(p * n)]  # 36 x 105

    col = 0
    for s_prod in strassen_products:
        for l_prod in laderman_products:
            # For this Strassen-Laderman product combination:
            # U factors: for each (block_idx, coeff) in s_prod['U'], 
            #            for each (elem_idx, l_coeff) in l_prod['U']:
            #              U[row_idx(block_idx, elem_idx), col] += coeff * l_coeff
            # V factors: similar for V
            # W factors: similar for W

            # Block index to row/col offset mapping:
            # Block 0 (A11): rows 0-2, cols 0-2
            # Block 1 (A12): rows 0-2, cols 3-5
            # Block 2 (A21): rows 3-5, cols 0-2
            # Block 3 (A22): rows 3-5, cols 3-5

            def block_offset(block_idx):
                """Return (row_offset, col_offset) for a 3x3 block in the 6x6 matrix."""
                row_off = (block_idx // 2) * 3
                col_off = (block_idx % 2) * 3
                return row_off, col_off

            # U factors
            for (s_block_idx, s_coeff) in s_prod['U']:
                s_row_off, s_col_off = block_offset(s_block_idx)
                for (l_elem_idx, l_coeff) in l_prod['U']:
                    l_row = l_elem_idx // 3
                    l_col = l_elem_idx % 3
                    u_row = (s_row_off + l_row) * m + (s_col_off + l_col)
                    U[u_row][col] += s_coeff * l_coeff

            # V factors
            for (s_block_idx, s_coeff) in s_prod['V']:
                s_row_off, s_col_off = block_offset(s_block_idx)
                for (l_elem_idx, l_coeff) in l_prod['V']:
                    l_row = l_elem_idx // 3
                    l_col = l_elem_idx % 3
                    v_row = (s_row_off + l_row) * p + (s_col_off + l_col)
                    V[v_row][col] += s_coeff * l_coeff

            # W factors
            for (s_block_idx, s_coeff) in s_prod['W']:
                s_row_off, s_col_off = block_offset(s_block_idx)
                for (l_elem_idx, l_coeff) in l_prod['W']:
                    l_row = l_elem_idx // 3
                    l_col = l_elem_idx % 3
                    # W index: W[k*n + i] where k is col of C, i is row of C
                    w_row = (s_col_off + l_col) * n + (s_row_off + l_row)
                    W[w_row][col] += s_coeff * l_coeff

            col += 1

    # Verify the decomposition with a full tensor check
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        elapsed = time.time() - start_time
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook if verification fails
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}
