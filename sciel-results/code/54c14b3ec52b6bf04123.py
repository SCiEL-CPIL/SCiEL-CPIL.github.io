"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition:
    # M_{2,4,8} -> 4 * M_{2,2,4} -> 8 * M_{2,2,2} (Strassen, rank 7)
    # Total rank = 8 * 7 = 56
    #
    # Structure:
    # A is 2x4, split into A_L (2x2), A_R (2x2)
    # B is 4x8, split into B_T (2x4), B_B (2x4)
    # C = A_L * B_T + A_R * B_B  (each term is 2x4)
    #
    # Each 2x4 product (2x2 * 2x4) uses 2 Strassen 2x2x2 multiplications:
    #   For M_{2,2,4}: B_2x4 = [B_L 2x2 | B_R 2x2]
    #   Result = [A*B_L | A*B_R]
    #   A*B_L uses one Strassen, A*B_R uses another Strassen
    #
    # So total 4 terms * 2 Strassen each = 8 Strassen multiplications
    # Each Strassen has rank 7, total rank = 56
    #
    # Strassen for 2x2 * 2x2:
    #   M1 = (A11+A22)*(B11+B22)
    #   M2 = (A21+A22)*B11
    #   M3 = A11*(B12-B22)
    #   M4 = A22*(B21-B11)
    #   M5 = (A11+A12)*B22
    #   M6 = (A21-A11)*(B11+B12)
    #   M7 = (A12-A22)*(B21+B22)
    #   C11 = M1+M4-M5+M7
    #   C12 = M3+M5
    #   C21 = M2+M4
    #   C22 = M1-M2+M3+M6
    #
    # Indexing:
    # A row index: i * m + j where i in [0,2), j in [0,4)
    # B row index: j * p + k where j in [0,4), k in [0,8)
    # W index: k * n + i where k in [0,8), i in [0,2)
    #
    # For Strassen on sub-matrix with A block starting at (i_off, a_off) and B block starting at (b_off, k_off):
    # A elements are at positions: (i_off + a_row) * 4 + (a_off + a_col) in U
    # B elements are at positions: (b_off + b_row) * 8 + (k_off + b_col) in V
    # W elements are at positions: (k_off + c_col) * 2 + (i_off + c_row) in W

    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56

    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    # Strassen coefficients for each of the 7 products:
    # Each entry is (coeff_A, coeff_B) for the two factors
    # Strassen products:
    # P1: (A11+A22) * (B11+B22)
    # P2: (A21+A22) * B11
    # P3: A11 * (B12-B22)
    # P4: A22 * (B21-B11)
    # P5: (A11+A12) * B22
    # P6: (A21-A11) * (B11+B12)
    # P7: (A12-A22) * (B21+B22)

    # For each Strassen product, we need to define:
    # - Which A elements with which coefficients
    # - Which B elements with which coefficients
    # - Which C elements with which coefficients

    # Strassen formula details:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b01-b11)
    # P4 = a11*(b10-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b01)
    # P7 = (a01-a11)*(b10+b11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # For each Strassen instance, define the 7 products
    # Each product has a list of (A_element_index, coeff) and (B_element_index, coeff)
    # A_element_index: which of the 4 elements (0=a00, 1=a01, 2=a10, 3=a11)
    # B_element_index: which of the 4 elements (0=b00, 1=b01, 2=b10, 3=b11)

    # The 7 Strassen products with their factor components
    strassen_products = [
        # (A factors: [(elem_idx, coeff)], B factors: [(elem_idx, coeff)])
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7
    ]

    # C reconstruction: C_elem = sum of P_products with coefficients
    # C00: P1 + P4 - P5 + P7 -> indices [0, 3, -4, 6]
    # C01: P3 + P5 -> indices [2, 4]
    # C10: P2 + P4 -> indices [1, 3]
    # C11: P1 - P2 + P3 + P6 -> indices [0, -1, 2, 5]
    c_reconstruction = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        [(2, 1), (4, 1)],                   # C01
        [(1, 1), (3, 1)],                   # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],  # C11
    ]

    # Now we have 8 Strassen instances:
    # Instance 0: A_L * B_TL (A_L is rows 0-1, cols 0-1; B_TL is rows 0-1, cols 0-1)
    # Instance 1: A_L * B_TR (A_L same; B_TR is rows 0-1, cols 2-3)
    # Instance 2: A_R * B_BL (A_R is rows 0-1, cols 2-3; B_BL is rows 2-3, cols 0-1)
    # Instance 3: A_R * B_BR (A_R same; B_BR is rows 2-3, cols 2-3)
    #
    # Wait, let me re-examine the block structure.
    # A is 2x4: A = [A_L | A_R] where A_L is 2x2 (cols 0-1), A_R is 2x2 (cols 2-3)
    # B is 4x8: B = [[B00, B01], [B10, B11]] where each is 2x4
    #   B00 is rows 0-1, cols 0-3
    #   B01 is rows 0-1, cols 4-7
    #   B10 is rows 2-3, cols 0-3
    #   B11 is rows 2-3, cols 4-7
    #
    # C = A * B = A_L * B00 + A_R * B10  (left half, cols 0-3)
    #           A_L * B01 + A_R * B11  (right half, cols 4-7)
    #
    # Each A_L * Bxx is 2x2 * 2x4 = M_{2,2,4}, which uses 2 Strassen instances
    # (one for left 2x2 part of Bxx, one for right 2x2 part of Bxx)
    #
    # So:
    # Instance 0: A_L * (B00 left 2x2) -> contributes to C cols 0-1
    # Instance 1: A_L * (B00 right 2x2) -> contributes to C cols 2-3
    # Instance 2: A_R * (B10 left 2x2) -> contributes to C cols 0-1
    # Instance 3: A_R * (B10 right 2x2) -> contributes to C cols 2-3
    # Instance 4: A_L * (B01 left 2x2) -> contributes to C cols 4-5
    # Instance 5: A_L * (B01 right 2x2) -> contributes to C cols 6-7
    # Instance 6: A_R * (B11 left 2x2) -> contributes to C cols 4-5
    # Instance 7: A_R * (B11 right 2x2) -> contributes to C cols 6-7

    # Define the 8 instances:
    # (a_row_offset, a_col_offset, b_row_offset, b_col_offset)
    # a_row_offset: which row of A (0 or 1, since A is 2 rows) - always 0 for the 2x2 block
    # Actually, A_L uses all 2 rows, cols 0-1; A_R uses all 2 rows, cols 2-3
    # So a_col_offset is 0 for A_L, 2 for A_R
    # b_row_offset: 0 for top 2 rows, 2 for bottom 2 rows
    # b_col_offset: depends on which quarter

    instances = [
        (0, 0, 0),  # A_L, B00 left  (b_col=0)
        (0, 0, 2),  # A_L, B00 right (b_col=2)
        (2, 0, 0),  # A_R, B10 left  (b_col=0)
        (2, 0, 2),  # A_R, B10 right (b_col=2)
        (0, 4, 0),  # A_L, B01 left  (b_col=0)
        (0, 4, 2),  # A_L, B01 right (b_col=2)
        (2, 4, 0),  # A_R, B11 left  (b_col=0)
        (2, 4, 2),  # A_R, B11 right (b_col=2)
    ]
    # instance = (a_col_off, b_row_off, b_col_off)
    # a_row_off is always 0 (both A_L and A_R use rows 0-1)
    # b_col_off is 0 or 2 within the 4-col block
    # But wait, B01 is cols 4-7, so b_col_off should be 4 or 6
    # Let me redefine:
    # instance = (a_col_off, b_row_off, b_col_off_global)
    # where b_col_off_global is 0, 2, 4, or 6

    # Wait, B is 4x8. Rows are 0-3. B00 is rows 0-1, B01 is rows 0-1 cols 4-7
    # I confused myself. Let me restart the block structure.

    # Correct structure:
    # A: 2x4 = [A_L | A_R], A_L = A[:, 0:2] (2x2), A_R = A[:, 2:4] (2x2)
    # B: 4x8 = [[B_TL, B_TR], [B_BL, B_BR]]
    #   B_TL = B[0:2, 0:4] (2x4), B_TR = B[0:2, 4:8] (2x4)
    #   B_BL = B[2:4, 0:4] (2x4), B_BR = B[2:4, 4:8] (2x4)
    # C = A_L * B_TL + A_R * B_BL  (cols 0-3)
    #   A_L * B_TR + A_R * B_BR  (cols 4-7)
    #
    # Each 2x2 * 2x4 uses 2 Strassen:
    #   A_L * B_TL: A_L * B_TL[:, 0:2] + A_L * B_TL[:, 2:4]
    #   etc.

    # So 8 Strassen instances:
    # (a_col_off, b_row_off, b_col_off)
    # a_col_off: 0 (A_L) or 2 (A_R)
    # b_row_off: 0 (top) or 2 (bottom)
    # b_col_off: 0, 2, 4, 6 (within the 8 columns)

    # For C cols 0-3 (left half of B):
    #   A_L * B[0:2, 0:2] -> (0, 0, 0)
    #   A_L * B[0:2, 2:4] -> (0, 0, 2)
    #   A_R * B[2:4, 0:2] -> (2, 2, 0)
    #   A_R * B[2:4, 2:4] -> (2, 2, 2)
    # For C cols 4-7 (right half of B):
    #   A_L * B[0:2, 4:6] -> (0, 0, 4)
    #   A_L * B[0:2, 6:8] -> (0, 0, 6)
    #   A_R * B[2:4, 4:6] -> (2, 2, 4)
    #   A_R * B[2:4, 6:8] -> (2, 2, 6)

    instances = [
        (0, 0, 0),
        (0, 0, 2),
        (2, 2, 0),
        (2, 2, 2),
        (0, 0, 4),
        (0, 0, 6),
        (2, 2, 4),
        (2, 2, 6),
    ]

    col = 0
    for inst_idx, (a_col_off, b_row_off, b_col_off) in enumerate(instances):
        for prod_idx in range(7):
            a_factors, b_factors = strassen_products[prod_idx]
            for a_elem, a_coeff in a_factors:
                # A element is at row (a_row=0 or 1), col (a_col_off + a_col)
                # a_elem: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                a_row = a_elem // 2
                a_col = a_elem % 2
                u_idx = a_row * m + (a_col_off + a_col)  # a_row * 4 + (a_col_off + a_col)
                U[u_idx][col] += a_coeff
            for b_elem, b_coeff in b_factors:
                # B element is at row (b_row_off + b_row), col (b_col_off + b_col)
                b_row = b_elem // 2
                b_col = b_elem % 2
                v_idx = (b_row_off + b_row) * p + (b_col_off + b_col)  # (b_row_off + b_row) * 8 + (b_col_off + b_col)
                V[v_idx][col] += b_coeff
            col += 1

    # Now add the W contributions from C reconstruction
    # For each instance, the 7 products contribute to the 4 C elements
    col = 0
    for inst_idx, (a_col_off, b_row_off, b_col_off) in enumerate(instances):
        for c_elem in range(4):
            # c_elem: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            c_row = c_elem // 2
            c_col = c_elem % 2
            # The C element is at row c_row (0 or 1), col (b_col_off + c_col)
            # W index: (b_col_off + c_col) * n + c_row = (b_col_off + c_col) * 2 + c_row
            w_idx = (b_col_off + c_col) * n + c_row
            for prod_idx, coeff in c_reconstruction[c_elem]:
                W[w_idx][col + prod_idx] += coeff
        col += 7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
