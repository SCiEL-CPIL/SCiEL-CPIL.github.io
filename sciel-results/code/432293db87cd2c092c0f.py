"""Matrix multiplication 2x4x5 - block decomposition with Strassen sub-algorithms."""

def construct():
    """
    Rank-36 exact tensor decomposition for 2x4 x 4x5 matrix multiplication.
    
    Structure:
      A (2x4) = [A_L | A_R]  where A_L, A_R are 2x2
      B (4x5) = [B_top; B_bot] where B_top, B_bot are 2x5
      C = A_L * B_top + A_R * B_bot
      
    Each 2x2x5 multiplication splits B's columns into (0,1), (2,3), (4):
      - Strassen (rank 7) for 2x2x2 parts
      - Schoolbook (rank 4) for 2x2x1 part
    Total rank: 2 * (7 + 7 + 4) = 36
    
    Index conventions:
      U[i*m + j, c] for i in [0,1], j in [0,3]
      V[j*p + k, c] for j in [0,3], k in [0,4]
      W[k*n + i, c] for k in [0,4], i in [0,1]
    """
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0.0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 36
    
    # Strassen coefficients for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11) B01
    # M3 = A00 B10
    # M4 = (A00+A01)(B11-B10)
    # M5 = (A10+A00)(B00-B01)
    # M6 = (A11-A10)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M5
    
    # For a 2x2x2 block with A rows (a0,a1), B rows (b0,b1), B cols (c0,c1):
    # U entries (A coefficients):
    #   M1: A00=+1, A11=+1
    #   M2: A10=+1, A11=+1
    #   M3: A00=+1
    #   M4: A00=+1, A01=+1
    #   M5: A10=+1, A00=+1
    #   M6: A11=+1, A10=-1
    #   M7: A01=+1, A11=-1
    #
    # V entries (B coefficients):
    #   M1: B00=+1, B11=+1
    #   M2: B01=+1
    #   M3: B10=+1
    #   M4: B11=+1, B10=-1
    #   M5: B00=+1, B01=-1
    #   M6: B00=+1, B01=+1
    #   M7: B10=+1, B11=+1
    #
    # W entries (C coefficients):
    #   M1: C00=+1, C11=+1
    #   M2: C10=+1, C11=-1
    #   M3: C01=+1, C11=+1
    #   M4: C00=+1
    #   M5: C00=-1, C11=+1
    #   M6: C10=+1
    #   M7: C00=-1, C01=+1
    
    def add_strassen_block(A_row_base, B_row_base, B_col_base, c_offset):
        """Add Strassen decomposition for 2x2x2 block."""
        # A rows: A_row_base, A_row_base+1
        # B rows: B_row_base, B_row_base+1
        # B cols: B_col_base, B_col_base+1
        # C output: rows 0,1 of the 2x2 result at cols B_col_base, B_col_base+1
        
        # M1: (A00+A11)(B00+B11)
        U[A_row_base][c_offset + 0] += 1.0
        U[A_row_base + 1][c_offset + 0] += 1.0
        V[B_row_base * p + B_col_base][c_offset + 0] += 1.0
        V[(B_row_base + 1) * p + (B_col_base + 1)][c_offset + 0] += 1.0
        W[B_col_base * n + 0][c_offset + 0] += 1.0   # C00
        W[(B_col_base + 1) * n + 1][c_offset + 0] += 1.0  # C11
        
        # M2: (A10+A11) B01
        U[A_row_base + 1][c_offset + 1] += 1.0
        V[B_row_base * p + (B_col_base + 1)][c_offset + 1] += 1.0
        W[B_col_base * n + 1][c_offset + 1] += 1.0   # C10
        W[(B_col_base + 1) * n + 1][c_offset + 1] += -1.0  # C11
        
        # M3: A00 B10
        U[A_row_base][c_offset + 2] += 1.0
        V[(B_row_base + 1) * p + B_col_base][c_offset + 2] += 1.0
        W[(B_col_base + 1) * n + 0][c_offset + 2] += 1.0  # C01
        W[(B_col_base + 1) * n + 1][c_offset + 2] += 1.0  # C11
        
        # M4: (A00+A01)(B11-B10)
        U[A_row_base][c_offset + 3] += 1.0
        V[(B_row_base + 1) * p + (B_col_base + 1)][c_offset + 3] += 1.0
        V[(B_row_base + 1) * p + B_col_base][c_offset + 3] += -1.0
        W[B_col_base * n + 0][c_offset + 3] += 1.0   # C00
        
        # M5: (A10+A00)(B00-B01)
        U[A_row_base + 1][c_offset + 4] += 1.0
        U[A_row_base][c_offset + 4] += 1.0
        V[B_row_base * p + B_col_base][c_offset + 4] += 1.0
        V[B_row_base * p + (B_col_base + 1)][c_offset + 4] += -1.0
        W[B_col_base * n + 0][c_offset + 4] += -1.0  # C00
        W[(B_col_base + 1) * n + 1][c_offset + 4] += 1.0  # C11
        
        # M6: (A11-A10)(B00+B01)
        U[A_row_base + 1][c_offset + 5] += 1.0
        U[A_row_base + 1][c_offset + 5] += 0.0  # A11=+1
        # Actually A11 is row 1, A10 is row 1 col 0... wait
        # A is 2x2: A00=(row0,col0), A01=(row0,col1), A10=(row1,col0), A11=(row1,col1)
        # In our U indexing: U[row, c] where row = i*m+j
        # For a 2x2 block at A rows (r0, r1) and cols (c0, c1):
        #   A00 -> U[r0*2+c0]... no, U is indexed by [i*m+j]
        # Let me reconsider.
        
        # Actually, let me redo this more carefully.
        # U has shape (n*m, rank) = (8, 36)
        # U[i*m + j, c] where i is row of A (0 or 1), j is col of A (0,1,2,3)
        # For block with A rows i0, i1 and A cols j0, j1:
        #   A00 = U[i0*m + j0]
        #   A01 = U[i0*m + j1]
        #   A10 = U[i1*m + j0]
        #   A11 = U[i1*m + j1]
        
        # Let me restart with proper indexing.
        pass
    
    # Let me rewrite with cleaner indexing
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen(A_i0, A_j0, B_j0, B_k0, c_off):
        """
        Add Strassen for 2x2x2 block.
        A: rows i0, i0+1; cols A_j0, A_j0+1
        B: rows B_j0, B_j0+1; cols B_k0, B_k0+1
        C: rows i0, i0+1; cols B_k0, B_k0+1
        """
        # A00=U[i0*m+A_j0], A01=U[i0*m+A_j0+1], A10=U[(i0+1)*m+A_j0], A11=U[(i0+1)*m+A_j0+1]
        # B00=V[B_j0*p+B_k0], B01=V[B_j0*p+B_k0+1], B10=V[(B_j0+1)*p+B_k0], B11=V[(B_j0+1)*p+B_k0+1]
        # C00=W[B_k0*n+i0], C01=W[(B_k0+1)*n+i0], C10=W[B_k0*n+(i0+1)], C11=W[(B_k0+1)*n+(i0+1)]
        
        a00 = i0 * m + A_j0
        a01 = i0 * m + A_j0 + 1
        a10 = (i0 + 1) * m + A_j0
        a11 = (i0 + 1) * m + A_j0 + 1
        
        b00 = B_j0 * p + B_k0
        b01 = B_j0 * p + B_k0 + 1
        b10 = (B_j0 + 1) * p + B_k0
        b11 = (B_j0 + 1) * p + B_k0 + 1
        
        c00 = B_k0 * n + i0
        c01 = (B_k0 + 1) * n + i0
        c10 = B_k0 * n + (i0 + 1)
        c11 = (B_k0 + 1) * n + (i0 + 1)
        
        # M1: (A00+A11)(B00+B11)
        U[a00][c_off + 0] += 1.0
        U[a11][c_off + 0] += 1.0
        V[b00][c_off + 0] += 1.0
        V[b11][c_off + 0] += 1.0
        W[c00][c_off + 0] += 1.0
        W[c11][c_off + 0] += 1.0
        
        # M2: (A10+A11) B01
        U[a10][c_off + 1] += 1.0
        U[a11][c_off + 1] += 1.0
        V[b01][c_off + 1] += 1.0
        W[c10][c_off + 1] += 1.0
        W[c11][c_off + 1] += -1.0
        
        # M3: A00 B10
        U[a00][c_off + 2] += 1.0
        V[b10][c_off + 2] += 1.0
        W[c01][c_off + 2] += 1.0
        W[c11][c_off + 2] += 1.0
        
        # M4: (A00+A01)(B11-B10)
        U[a00][c_off + 3] += 1.0
        U[a01][c_off + 3] += 1.0
        V[b11][c_off + 3] += 1.0
        V[b10][c_off + 3] += -1.0
        W[c00][c_off + 3] += 1.0
        
        # M5: (A10+A00)(B00-B01)
        U[a10][c_off + 4] += 1.0
        U[a00][c_off + 4] += 1.0
        V[b00][c_off + 4] += 1.0
        V[b01][c_off + 4] += -1.0
        W[c00][c_off + 4] += -1.0
        W[c11][c_off + 4] += 1.0
        
        # M6: (A11-A10)(B00+B01)
        U[a11][c_off + 5] += 1.0
        U[a10][c_off + 5] += -1.0
        V[b00][c_off + 5] += 1.0
        V[b01][c_off + 5] += 1.0
        W[c10][c_off + 5] += 1.0
        
        # M7: (A01-A11)(B10+B11)
        U[a01][c_off + 6] += 1.0
        U[a11][c_off + 6] += -1.0
        V[b10][c_off + 6] += 1.0
        V[b11][c_off + 6] += 1.0
        W[c00][c_off + 6] += -1.0
        W[c01][c_off + 6] += 1.0
    
    def add_schoolbook_2x2x1(A_i0, A_j0, B_j0, B_k, c_off):
        """
        Add schoolbook for 2x2x1 block.
        A: rows i0, i0+1; col A_j0
        B: rows B_j0, B_j0+1; col B_k
        C: rows i0, i0+1; col B_k
        """
        a0 = i0 * m + A_j0
        a1 = (i0 + 1) * m + A_j0
        b0 = B_j0 * p + B_k
        b1 = (B_j0 + 1) * p + B_k
        c0 = B_k * n + i0
        c1 = B_k * n + (i0 + 1)
        
        # M1: A0 * B0
        U[a0][c_off + 0] += 1.0
        V[b0][c_off + 0] += 1.0
        W[c0][c_off + 0] += 1.0
        
        # M2: A0 * B1
        U[a0][c_off + 1] += 1.0
        V[b1][c_off + 1] += 1.0
        W[c0][c_off + 1] += 1.0
        
        # M3: A1 * B0
        U[a1][c_off + 2] += 1.0
        V[b0][c_off + 2] += 1.0
        W[c1][c_off + 2] += 1.0
        
        # M4: A1 * B1
        U[a1][c_off + 3] += 1.0
        V[b1][c_off + 3] += 1.0
        W[c1][c_off + 3] += 1.0
    
    # Block 1: A_L (cols 0,1) * B_top (rows 0,1)
    # Sub-block 1.1: cols 0,1 -> Strassen
    add_strassen(0, 0, 0, 0, 0)       # c_off = 0
    # Sub-block 1.2: cols 2,3 -> Strassen
    add_strassen(0, 0, 0, 2, 7)       # c_off = 7
    # Sub-block 1.3: col 4 -> Schoolbook
    add_schoolbook_2x2x1(0, 0, 0, 4, 14)  # c_off = 14
    
    # Block 2: A_R (cols 2,3) * B_bot (rows 2,3)
    # Sub-block 2.1: cols 0,1 -> Strassen
    add_strassen(0, 2, 2, 0, 18)      # c_off = 18
    # Sub-block 2.2: cols 2,3 -> Strassen
    add_strassen(0, 2, 2, 2, 25)      # c_off = 25
    # Sub-block 2.3: col 4 -> Schoolbook
    add_schoolbook_2x2x1(0, 2, 2, 4, 32)  # c_off = 32
    
    return {"U": U, "V": V, "W": W}
