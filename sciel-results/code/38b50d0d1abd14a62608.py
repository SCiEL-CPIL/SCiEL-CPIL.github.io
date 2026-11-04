"""Matrix multiplication 2x4x4 - block Strassen decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 decomposition for 2x4 by 4x4 matrix multiplication.
    
    We partition A (2x4) into two 2x2 blocks [A1, A2] and B (4x4) into four 2x2 blocks.
    The output C (2x4) is [C1, C2] where:
      C1 = A1*B11 + A2*B21
      C2 = A1*B12 + A2*B22
    
    Each 2x2x2 multiplication uses Strassen's 7-multiplication algorithm.
    We apply Strassen independently to each of the 4 sub-problems,
    sharing the structure but keeping them separate.
    
    Total rank: 4 * 7 = 28
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen formulas for 2x2 * 2x2 = 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_block(u_offset, v_offset, w_offset, 
                           a_rows, b_rows, c_rows, c_cols,
                           b_block_rows, b_block_cols):
        """
        Add a Strassen block for multiplying a 2x2 block of A by a 2x2 block of B.
        
        a_rows: [r1, r2] - the two rows in A (0-indexed into the 2x4 matrix)
        b_rows: [r1, r2] - the two rows in B (0-indexed into the 4x4 matrix)
        c_rows: [r1, r2] - the two rows in C (0-indexed into the 2x4 output)
        c_cols: [c1, c2] - the two columns in C
        b_block_rows: [r1, r2] - which 2x2 block rows in B (for indexing)
        b_block_cols: [c1, c2] - which 2x2 block cols in B (for indexing)
        """
        # Strassen's 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        col = 0
        U[a_rows[0]][u_offset + col] = 1  # A11
        U[a_rows[1]][u_offset + col] = 1  # A22
        V[b_rows[0]][v_offset + col] = 1  # B11
        V[b_rows[1]][v_offset + col] = 1  # B22
        W[c_rows[0], c_cols[0]][w_offset + col] = 1   # C11 += M1
        W[c_rows[1], c_cols[1]][w_offset + col] = 1   # C22 += M1
        col += 1
        
        # M2 = (A21+A22)B11
        U[a_rows[1]][u_offset + col] = 1  # A21 (row 2, col 1 of A block)
        # Wait, A21 means row 2, col 1 of the 2x2 A block
        # In the full A matrix (2x4), a_rows[1] is the second row of the 2x2 block
        # But we need to be careful about column indexing
        # Let me reconsider the indexing...
        
    # Let me redo this with clearer indexing.
    # A is 2x4: A[i][j] for i in 0,1 and j in 0,1,2,3
    # B is 4x4: B[i][j] for i in 0,1,2,3 and j in 0,1,2,3
    # C is 2x4: C[i][j] for i in 0,1 and j in 0,1,2,3
    #
    # Partition:
    # A = [A1 | A2] where A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B = [[B11, B12], [B21, B22]] where B11 = B[0:2, 0:2], etc.
    # C = [C1 | C2] where C1 = C[:, 0:2], C2 = C[:, 2:4]
    #
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(u_base, v_base, w_base, 
                     a_col_offset, b_row_offset, b_col_offset,
                     c_col_offset):
        """
        Add Strassen block for A[:, a_col_offset:a_col_offset+2] * 
                                B[b_row_offset:b_row_offset+2, b_col_offset:b_col_offset+2]
        contributing to C[:, c_col_offset:c_col_offset+2]
        
        a_col_offset: starting column in A (0 or 2)
        b_row_offset: starting row in B (0 or 2)
        b_col_offset: starting col in B (0 or 2)
        c_col_offset: starting col in C (0 or 2)
        """
        # A indices: rows 0,1; cols a_col_offset, a_col_offset+1
        # B indices: rows b_row_offset, b_row_offset+1; cols b_col_offset, b_col_offset+1
        # C indices: rows 0,1; cols c_col_offset, c_col_offset+1
        
        a11_row, a11_col = 0, a_col_offset
        a12_row, a12_col = 0, a_col_offset + 1
        a21_row, a21_col = 1, a_col_offset
        a22_row, a22_col = 1, a_col_offset + 1
        
        b11_row, b11_col = b_row_offset, b_col_offset
        b12_row, b12_col = b_row_offset, b_col_offset + 1
        b21_row, b21_col = b_row_offset + 1, b_col_offset
        b22_row, b22_col = b_row_offset + 1, b_col_offset + 1
        
        c11_row, c11_col = 0, c_col_offset
        c12_row, c12_col = 0, c_col_offset + 1
        c21_row, c21_col = 1, c_col_offset
        c22_row, c22_col = 1, c_col_offset + 1
        
        # M1 = (A11+A22)(B11+B22)
        col = 0
        U[a11_row * m + a11_col][u_base + col] += 1
        U[a22_row * m + a22_col][u_base + col] += 1
        V[b11_row * p + b11_col][v_base + col] += 1
        V[b22_row * p + b22_col][v_base + col] += 1
        W[c11_row * n + c11_col][w_base + col] += 1
        W[c22_row * n + c22_col][w_base + col] += 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[a21_row * m + a21_col][u_base + col] += 1
        U[a22_row * m + a22_col][u_base + col] += 1
        V[b11_row * p + b11_col][v_base + col] += 1
        W[c21_row * n + c21_col][w_base + col] += 1
        W[c22_row * n + c22_col][w_base + col] -= 1
        col += 1
        
        # M3 = A11(B21-B22)
        U[a11_row * m + a11_col][u_base + col] += 1
        V[b21_row * p + b21_col][v_base + col] += 1
        V[b22_row * p + b22_col][v_base + col] -= 1
        W[c12_row * n + c12_col][w_base + col] += 1
        W[c22_row * n + c22_col][w_base + col] += 1
        col += 1
        
        # M4 = A22(B12-B11)
        U[a22_row * m + a22_col][u_base + col] += 1
        V[b12_row * p + b12_col][v_base + col] += 1
        V[b11_row * p + b11_col][v_base + col] -= 1
        W[c11_row * n + c11_col][w_base + col] += 1
        W[c21_row * n + c21_col][w_base + col] += 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[a11_row * m + a11_col][u_base + col] += 1
        U[a12_row * m + a12_col][u_base + col] += 1
        V[b22_row * p + b22_col][v_base + col] += 1
        W[c11_row * n + c11_col][w_base + col] -= 1
        W[c12_row * n + c12_col][w_base + col] += 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a21_row * m + a21_col][u_base + col] += 1
        U[a11_row * m + a11_col][u_base + col] -= 1
        V[b11_row * p + b11_col][v_base + col] += 1
        V[b12_row * p + b12_col][v_base + col] += 1
        W[c22_row * n + c22_col][w_base + col] += 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a12_row * m + a12_col][u_base + col] += 1
        U[a22_row * m + a22_col][u_base + col] -= 1
        V[b21_row * p + b21_col][v_base + col] += 1
        V[b22_row * p + b22_col][v_base + col] += 1
        W[c11_row * n + c11_col][w_base + col] += 1
        col += 1
    
    # Four Strassen blocks:
    # 1. A1 * B11 -> C1
    add_strassen(0, 0, 0, a_col_offset=0, b_row_offset=0, b_col_offset=0, c_col_offset=0)
    # 2. A2 * B21 -> C1
    add_strassen(7, 7, 7, a_col_offset=2, b_row_offset=2, b_col_offset=0, c_col_offset=0)
    # 3. A1 * B12 -> C2
    add_strassen(14, 14, 14, a_col_offset=0, b_row_offset=0, b_col_offset=2, c_col_offset=2)
    # 4. A2 * B22 -> C2
    add_strassen(21, 21, 21, a_col_offset=2, b_row_offset=2, b_col_offset=2, c_col_offset=2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
