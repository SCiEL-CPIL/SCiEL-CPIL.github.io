"""Program for rank-minimal exact tensor decomposition of 2x4x4 matrix multiplication."""

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 decomposition for 2x4 x 4x4 matrix multiplication.
    
    The decomposition uses a block-wise Strassen approach:
    - Split the 2x4 matrix A into two 2x2 blocks: A_left (cols 0-1), A_right (cols 2-3)
    - Split the 4x4 matrix B into four 2x2 blocks: B_LL, B_LR, B_RL, B_RR
    - Split the 2x4 result C into two 2x2 blocks: C_left (cols 0-1), C_right (cols 2-3)
    
    For each of the 4 block multiplications (2x2 x 2x2 = 2x2), apply Strassen's 7-term algorithm.
    Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]  # (2*4, 28)
    V = [[0.0] * rank for _ in range(m * p)]  # (4*4, 28)
    W = [[0.0] * rank for _ in range(p * n)]  # (4*2, 28)
    
    # Strassen coefficients for 2x2x2 matrix multiplication
    # For block multiplication X(2x2) * Y(2x2) = Z(2x2):
    # M1 = (X00+X11)(Y00+Y11) -> Z00, Z11
    # M2 = (X10+X11)Y00       -> Z10, Z11
    # M3 = X00(Y01-Y11)       -> Z01
    # M4 = X11(Y10-Y00)       -> Z00, Z10
    # M5 = (X00+X01)Y11       -> Z01, Z11
    # M6 = (X10-X00)(Y00+Y01) -> Z11
    # M7 = (X01-X11)(Y10+Y11) -> Z00, Z01
    
    # X indices (rows, cols): X00=(0,0), X01=(0,1), X10=(1,0), X11=(1,1)
    # Y indices (rows, cols): Y00=(0,0), Y01=(0,1), Y10=(1,0), Y11=(1,1)
    # Z indices (rows, cols): Z00=(0,0), Z01=(0,1), Z10=(1,0), Z11=(1,1)
    
    # Define the 7 Strassen terms with their contributions to X, Y, Z
    # Each term: (x_coeffs, y_coeffs, z_coeffs)
    # x_coeffs: list of (row, col, coeff)
    # y_coeffs: list of (row, col, coeff)
    # z_coeffs: list of (row, col, coeff)
    
    strassen_terms = [
        # M1 = (X00+X11)(Y00+Y11)
        {
            'x': [(0, 0, 1), (1, 1, 1)],
            'y': [(0, 0, 1), (1, 1, 1)],
            'z': [(0, 0, 1), (1, 1, 1)]
        },
        # M2 = (X10+X11)Y00
        {
            'x': [(1, 0, 1), (1, 1, 1)],
            'y': [(0, 0, 1)],
            'z': [(1, 0, 1), (1, 1, -1)]  # Z10 gets +1, Z11 gets... wait
        },
    ]
    
    # Let me be more careful with Strassen.
    # Standard Strassen:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Redo with 1-indexed notation mapped to 0-indexed:
    # A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    strassen_terms = [
        # M1 = (A11+A22)(B11+B22) -> C11(+1), C22(+1)
        {
            'x': [(0, 0, 1), (1, 1, 1)],
            'y': [(0, 0, 1), (1, 1, 1)],
            'z': [(0, 0, 1), (1, 1, 1)]
        },
        # M2 = (A21+A22)B11 -> C21(+1), C22(-1)
        {
            'x': [(1, 0, 1), (1, 1, 1)],
            'y': [(0, 0, 1)],
            'z': [(1, 0, 1), (1, 1, -1)]
        },
        # M3 = A11(B12-B22) -> C12(+1), C22(+1)
        {
            'x': [(0, 0, 1)],
            'y': [(0, 1, 1), (1, 1, -1)],
            'z': [(0, 1, 1), (1, 1, 1)]
        },
        # M4 = A22(B21-B11) -> C11(+1), C21(+1)
        {
            'x': [(1, 1, 1)],
            'y': [(1, 0, 1), (0, 0, -1)],
            'z': [(0, 0, 1), (1, 0, 1)]
        },
        # M5 = (A11+A12)B22 -> C11(-1), C12(+1)
        {
            'x': [(0, 0, 1), (0, 1, 1)],
            'y': [(1, 1, 1)],
            'z': [(0, 0, -1), (0, 1, 1)]
        },
        # M6 = (A21-A11)(B11+B12) -> C22(+1)
        {
            'x': [(1, 0, 1), (0, 0, -1)],
            'y': [(0, 0, 1), (0, 1, 1)],
            'z': [(1, 1, 1)]
        },
        # M7 = (A12-A22)(B21+B22) -> C11(+1), C12(+1)
        {
            'x': [(0, 1, 1), (1, 1, -1)],
            'y': [(1, 0, 1), (1, 1, 1)],
            'z': [(0, 0, 1), (0, 1, 1)]
        },
    ]
    
    # Block definitions:
    # Block 0: A_left * B_LL -> C_left (cols 0-1 of C)
    #   A rows 0-1, A cols 0-1; B rows 0-1, B cols 0-1; C rows 0-1, C cols 0-1
    # Block 1: A_left * B_LR -> C_right (cols 2-3 of C)
    #   A rows 0-1, A cols 0-1; B rows 0-1, B cols 2-3; C rows 0-1, C cols 2-3
    # Block 2: A_right * B_RL -> C_left (cols 0-1 of C)
    #   A rows 0-1, A cols 2-3; B rows 2-3, B cols 0-1; C rows 0-1, C cols 0-1
    # Block 3: A_right * B_RR -> C_right (cols 2-3 of C)
    #   A rows 0-1, A cols 2-3; B rows 2-3, B cols 2-3; C rows 0-1, C cols 2-3
    
    # Block parameters: (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset)
    blocks = [
        (0, 0, 0, 0, 0, 0),    # Block 0
        (0, 0, 0, 2, 0, 2),    # Block 1
        (0, 2, 2, 0, 0, 0),    # Block 2
        (0, 2, 2, 2, 0, 2),    # Block 3
    ]
    
    col = 0
    for block_idx, (ar_off, ac_off, br_off, bc_off, cr_off, cc_off) in enumerate(blocks):
        for term_idx, term in enumerate(strassen_terms):
            # Add contributions to U (A matrix)
            for (xr, xc, coeff) in term['x']:
                a_row = ar_off + xr
                a_col = ac_off + xc
                U[a_row * m + a_col][col] += coeff
            
            # Add contributions to V (B matrix)
            for (yr, yc, coeff) in term['y']:
                b_row = br_off + yr
                b_col = bc_off + yc
                V[b_row * p + b_col][col] += coeff
            
            # Add contributions to W (C matrix, using trace convention W[k*n+i, r])
            for (zr, zc, coeff) in term['z']:
                c_row = cr_off + zr
                c_col = cc_off + zc
                # W index: k*n + i where k is col index, i is row index
                w_idx = c_col * n + c_row
                W[w_idx][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
