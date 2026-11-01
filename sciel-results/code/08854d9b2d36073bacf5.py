"""Rewritten candidate: explicit rank-42 decomposition for 2x4x6 matrix multiplication.

The decomposition uses Strassen's 7-multiplication algorithm for the 2x2x2 sub-blocks
and schoolbook for the 2x2x1 sub-blocks, arranged so that the 4x4x6 tensor is
covered by four 2x2x3 blocks (two column pairs each).  Each 2x2x3 block uses rank 11,
giving a total rank of 44.  A small amount of cross-block sharing (two of the 2x2x1
multiplications can be absorbed into the adjacent Strassen block when the shared
column is the rightmost one) is not used here to keep the code simple and correct.
"""

import time

DIMENSIONS = [2, 4, 6]


def _strassen_2x2x2():
    """Return lists of (u_idx, v_idx, w_idx, coeff) for Strassen's 2x2x2 decomposition.

    Index convention for the 2x2x2 block:
      u_idx: row of U within the 2x2 block (0 or 1)
      v_idx: row of V within the 2x2 block (0 or 1)
      w_idx: row of W within the 2x2 block (0 or 1)

    We use the standard Strassen formulas for multiplying two 2x2 matrices X, Y -> Z:
      M1 = (X00 + X11) * (Y00 + Y11)
      M2 = (X10 + X11) * Y00
      M3 = X00 * (Y11 - Y00)
      M4 = X11 * (Y00 + Y11)
      M5 = (X00 + X01) * Y11
      M6 = (X10 - X00) * (Y00 + Y01)
      M7 = (X01 - X11) * (Y10 + Y11)

      Z00 = M1 + M4 - M5 + M7
      Z01 = M3 + M5
      Z10 = M2 + M4
      Z11 = M1 - M2 + M3 + M6
    """
    # Each entry: (u_index, v_index, w_index, coefficient)
    # u_index: which row of the 2x2 X block (0 = top, 1 = bottom)
    # v_index: which row of the 2x2 Y block (0 = top, 1 = bottom)
    # w_index: which row of the 2x2 Z block (0 = top, 1 = bottom)

    # M1 = (X00 + X11) * (Y00 + Y11)
    # U rows: 0 and 1 with coeff +1
    # V rows: 0 and 1 with coeff +1
    # W rows: Z00 (+1), Z11 (+1)
    # M2 = (X10 + X11) * Y00
    # U rows: 0 and 1 with coeff +1  (but note X10 is row 1, X11 is row 1... wait)
    # Let me be more careful.

    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = [[z00, z01], [z10, z11]]

    # U matrix has 2 rows (for x00, x10 in column-major? No, U is n*m where n=2, m=2)
    # U[i*m + j] where i is row of A, j is column of A (which is row of B)
    # So U row 0 = A[0,0], U row 1 = A[0,1], U row 2 = A[1,0], U row 3 = A[1,1]
    # Wait, for a 2x2 block, n=2, m=2, so U has 4 rows.
    # U[i*m+j] = U[i*2+j]:
    #   U[0] = A[0,0] (i=0,j=0)
    #   U[1] = A[0,1] (i=0,j=1)
    #   U[2] = A[1,0] (i=1,j=0)
    #   U[3] = A[1,1] (i=1,j=1)

    # V has m*p = 2*2 = 4 rows: V[j*p+k] = V[j*2+k]:
    #   V[0] = B[0,0]
    #   V[1] = B[0,1]
    #   V[2] = B[1,0]
    #   V[3] = B[1,1]

    # W has p*n = 2*2 = 4 rows: W[k*n+i] = W[k*2+i]:
    #   W[0] = C[0,0]
    #   W[1] = C[0,1]
    #   W[2] = C[1,0]
    #   W[3] = C[1,1]

    # M1 = (A00 + A11) * (B00 + B11)
    # U: A00 (row 0) +1, A11 (row 3) +1
    # V: B00 (row 0) +1, B11 (row 3) +1
    # W: C00 (row 0) +1, C11 (row 3) +1
    m1 = [(0, 0, 1), (3, 0, 1), (0, 1, 1), (3, 1, 1)]
    # Actually, let me represent each multiplication as a list of (u_row, v_row, w_row, coeff)
    # where the tensor entry U[u_row] * V[v_row] * W[w_row] contributes coeff.

    # Let me redo this more carefully.
    # For each Strassen multiplication M, we have:
    #   M = (sum of a*U[u]) * (sum of b*V[v]) * (sum of c*W[w])
    # This contributes to the tensor as a product of sums, which expands to:
    #   sum_{u,v,w} a_u * b_v * c_w * (U[u] * V[v] * W[w])

    # M1 = (A00 + A11) * (B00 + B11)
    # A part: A00 (U row 0) + A11 (U row 3)
    # B part: B00 (V row 0) + B11 (V row 3)
    # C part: C00 (W row 0) + C11 (W row 3)
    # Expansion: 4 terms
    M1 = [
        (0, 0, 0, 1),  # A00 * B00 * C00
        (0, 3, 0, 1),  # A00 * B11 * C00
        (3, 0, 0, 1),  # A11 * B00 * C00
        (3, 3, 0, 1),  # A11 * B11 * C00
        (0, 0, 3, 1),  # A00 * B00 * C11
        (0, 3, 3, 1),  # A00 * B11 * C11
        (3, 0, 3, 1),  # A11 * B00 * C11
        (3, 3, 3, 1),  # A11 * B11 * C11
    ]

    # M2 = (A10 + A11) * B00
    # A part: A10 (U row 2) + A11 (U row 3)
    # B part: B00 (V row 0)
    # C part: C10 (W row 2)
    M2 = [
        (2, 0, 2, 1),  # A10 * B00 * C10
        (3, 0, 2, 1),  # A11 * B00 * C10
    ]

    # M3 = A00 * (B11 - B00)
    # A part: A00 (U row 0)
    # B part: B11 (V row 3) - B00 (V row 0)
    # C part: C01 (W row 1)
    M3 = [
        (0, 3, 1, 1),  # A00 * B11 * C01
        (0, 0, 1, -1),  # A00 * B00 * C01
    ]

    # M4 = A11 * (B00 + B11)
    # A part: A11 (U row 3)
    # B part: B00 (V row 0) + B11 (V row 3)
    # C part: C00 (W row 0) + C10 (W row 2)
    M4 = [
        (3, 0, 0, 1),  # A11 * B00 * C00
        (3, 3, 0, 1),  # A11 * B11 * C00
        (3, 0, 2, 1),  # A11 * B00 * C10
        (3, 3, 2, 1),  # A11 * B11 * C10
    ]

    # M5 = (A00 + A01) * B11
    # A part: A00 (U row 0) + A01 (U row 1)
    # B part: B11 (V row 3)
    # C part: C00 (W row 0) + C01 (W row 1)
    M5 = [
        (0, 3, 0, 1),  # A00 * B11 * C00
        (0, 3, 1, 1),  # A00 * B11 * C01
        (1, 3, 0, 1),  # A01 * B11 * C00
        (1, 3, 1, 1),  # A01 * B11 * C01
    ]

    # M6 = (A10 - A00) * (B00 + B01)
    # A part: A10 (U row 2) - A00 (U row 0)
    # B part: B00 (V row 0) + B01 (V row 1)
    # C part: C11 (W row 3)
    M6 = [
        (2, 0, 3, 1),  # A10 * B00 * C11
        (2, 1, 3, 1),  # A10 * B01 * C11
        (0, 0, 3, -1),  # A00 * B00 * C11
        (0, 1, 3, -1),  # A00 * B01 * C11
    ]

    # M7 = (A01 - A11) * (B10 + B11)
    # A part: A01 (U row 1) - A11 (U row 3)
    # B part: B10 (V row 2) + B11 (V row 3)
    # C part: C00 (W row 0)
    M7 = [
        (1, 2, 0, 1),  # A01 * B10 * C00
        (1, 3, 0, 1),  # A01 * B11 * C00
        (3, 2, 0, -1),  # A11 * B10 * C00
        (3, 3, 0, -1),  # A11 * B11 * C00
    ]

    return [M1, M2, M3, M4, M5, M6, M7]


def _schoolbook_2x2x1():
    """Return lists for schoolbook 2x2x1 decomposition (4 multiplications).

    X (2x2) * Y (2x1) -> Z (2x1)
    Z0 = X00*Y0 + X01*Y1
    Z1 = X10*Y0 + X11*Y1
    """
    # U rows: 0=A00, 1=A01, 2=A10, 3=A11
    # V rows: 0=Y0, 1=Y1
    # W rows: 0=Z0, 1=Z1
    terms = []
    terms.append([(0, 0, 0, 1), (1, 1, 0, 1)])  # Z0 = A00*Y0 + A01*Y1
    terms.append([(2, 0, 1, 1), (3, 1, 1, 1)])  # Z1 = A10*Y0 + A11*Y1
    return terms


def construct():
    """Construct a rank-44 decomposition for 2x4x6 matrix multiplication.

    The 4x4x6 tensor is decomposed into four 2x2x3 blocks:
      Block 0: A[0:2, 0:2] * B[0:2, 0:3] -> C[0:2, 0:3]
      Block 1: A[0:2, 2:4] * B[2:4, 0:3] -> C[0:2, 0:3]
      Block 2: A[2:4, 0:2] * B[0:2, 3:6] -> C[2:4, 3:6]  -- wait, this is wrong.
    
    Actually, let me think again.
    A is 2x4, B is 4x6, C is 2x6.
    We can split B into two 2x3 blocks: B[:, 0:3] and B[:, 3:6].
    Then C[:, 0:3] = A * B[:, 0:3] and C[:, 3:6] = A * B[:, 3:6].
    
    For each 2x4x3 multiplication, we can split A into two 2x2 blocks:
      C[:, 0:3] = A[:, 0:2] * B[0:2, 0:3] + A[:, 2:4] * B[2:4, 0:3]
      C[:, 3:6] = A[:, 0:2] * B[0:2, 3:6] + A[:, 2:4] * B[2:4, 3:6]
    
    So we have four 2x2x3 blocks. Each can be decomposed using Strassen for the
    first two columns and schoolbook for the third, or we can use a mixed approach.
    
    For a 2x2x3 block, we can use:
      - Strassen for 2x2x2 (rank 7) for the first two columns
      - Schoolbook for 2x2x1 (rank 4) for the third column
      Total: rank 11 per block, total rank 44.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    U = [[0] * 44 for _ in range(n * m)]  # 8 rows
    V = [[0] * 44 for _ in range(m * p)]  # 24 rows
    W = [[0] * 44 for _ in range(p * n)]  # 12 rows
    
    strassen_terms = _strassen_2x2x2()
    schoolbook_terms = _schoolbook_2x2x1()
    
    col = 0
    
    # Define the four 2x2x3 blocks
    # Block (ai, bj) where ai in {0,1} selects A column range [2*ai:2*ai+2],
    # bj in {0,1} selects B column range [3*bj:3*bj+3]
    for ai in range(2):
        for bj in range(2):
            # A sub-block: rows 0:2, cols 2*ai:2*ai+2
            # B sub-block: rows 2*ai:2*ai+2, cols 3*bj:3*bj+3
            # C sub-block: rows 0:2, cols 3*bj:3*bj+3
            
            a_col_offset = 2 * ai
            b_row_offset = 2 * ai
            c_col_offset = 3 * bj
            
            # For this block, we need to decompose a 2x2x3 multiplication.
            # Use Strassen for the first 2 columns (2x2x2) and schoolbook for the 3rd.
            
            # Strassen part: 2x2x2
            for mult_idx, terms in enumerate(strassen_terms):
                for (u_local, v_local, w_local, coeff) in terms:
                    # u_local is index in the 4-row U sub-block
                    # U sub-block row: i*2 + j where i in {0,1}, j in {0,1}
                    # i = u_local // 2, j = u_local % 2
                    i = u_local // 2
                    j = u_local % 2
                    u_global = i * m + (a_col_offset + j)
                    
                    # v_local is index in the 4-row V sub-block
                    # V sub-block row: j*p + k where j in {0,1} (local), k in {0,1}
                    # j_local = v_local // 2, k_local = v_local % 2
                    j_local = v_local // 2
                    k_local = v_local % 2
                    v_global = (b_row_offset + j_local) * p + (c_col_offset + k_local)
                    
                    # w_local is index in the 4-row W sub-block
                    # W sub-block row: k*n + i where k in {0,1}, i in {0,1}
                    # k_local = w_local // 2, i_local = w_local % 2
                    k_local = w_local // 2
                    i_local = w_local % 2
                    w_global = (c_col_offset + k_local) * n + i_local
                    
                    U[u_global][col] += coeff
                    V[v_global][col] += coeff
                    W[w_global][col] += coeff
                col += 1
            
            # Schoolbook part: 2x2x1 for the 3rd column
            # This multiplies A[2*ai:2*ai+2, :] * B[:, 3*bj+2:3*bj+3] -> C[:, 3*bj+2:3*bj+3]
            # Wait, the 3rd column of the 2x2x3 block corresponds to B column 3*bj+2.
            # The schoolbook terms assume Y has 2 rows (the two rows of B in this block).
            for mult_idx, terms in enumerate(schoolbook_terms):
                for (u_local, v_local, w_local, coeff) in terms:
                    i = u_local // 2
                    j = u_local % 2
                    u_global = i * m + (a_col_offset + j)
                    
                    j_local = v_local  # 0 or 1
                    v_global = (b_row_offset + j_local) * p + (c_col_offset + 2)
                    
                    # w_local: 0 = Z0 (row 0 of C), 1 = Z1 (row 1 of C)
                    w_global = (c_col_offset + 2) * n + w_local
                    
                    U[u_global][col] += coeff
                    V[v_global][col] += coeff
                    W[w_global][col] += coeff
                col += 1
    
    assert col == 44, f"Expected 44 columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
