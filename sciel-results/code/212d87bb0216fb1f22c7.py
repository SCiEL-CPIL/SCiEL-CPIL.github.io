"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    # C = A * B, A is n x m = 4x4, B is m x p = 4x6, C is n x p = 4x6
    # W convention: W[k*n + i, r] for output C[i,k]
    # U index: i*m + j for A[i,j]
    # V index: j*p + k for B[j,k]
    
    # Approach: Split B into two 4x3 blocks
    # B = [B1 | B2], each 4x3
    # C = A*B1 + A*B2
    # For each 4x4 x 4x3 multiplication, use a rank decomposition.
    # 
    # For 4x4 x 4x3: schoolbook is 4*4*3 = 48
    # But we can do better. Let's use Strassen for the 4x4 x 4x4 part and handle the 4x4 x 4x2 part.
    #
    # Better approach: Split into 4x4 x 4x4 (rank 49 via Strassen) + 4x4 x 4x2 (rank 32 schoolbook)
    # Total: 81
    
    rank = 81
    U = [[0] * rank for _ in range(n * m)]  # 16 x 81
    V = [[0] * rank for _ in range(m * p)]  # 24 x 81
    W = [[0] * rank for _ in range(p * n)]  # 24 x 81
    
    def add_strassen_4x4(U, V, W, col_offset, a_row_start, b_col_start, c_col_start):
        """Add Strassen decomposition for 4x4 x 4x4.
        a_row_start: starting row in A (0-based)
        b_col_start: starting col in B (0-based) 
        c_col_start: starting col in C (0-based)
        Returns next column offset.
        """
        # A is 4x4, B is 4x4, C is 4x4
        # Split into 2x2 blocks of 2x2
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        # C = [[C00, C01], [C10, C11]]
        
        # Strassen:
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)B00        -> C10 += M2, C11 += M2
        # M3 = A00(B01-B11)        -> C01 += M3, C11 -= M3
        # M4 = A11(B10-B00)        -> C00 -= M4, C10 += M4
        # M5 = (A00+A01)B11        -> C00 += M5, C01 += M5
        # M6 = (A10-A00)(B00+B01)  -> C11 -= M6
        # M7 = (A01-A11)(B10+B11)  -> C00 -= M7
        
        col = col_offset
        
        # For each Strassen product, we need to compute a 2x2 x 2x2 product
        # using the 7-term Strassen decomposition at the sub-block level.
        
        def add_2x2_product(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, c_coeffs):
            """Add a 2x2 x 2x2 Strassen decomposition.
            a_rows: list of 2 row indices for A sub-block
            a_cols: list of 2 col indices for A sub-block
            b_rows: list of 2 row indices for B sub-block
            b_cols: list of 2 col indices for B sub-block
            c_rows: list of 2 row indices for C sub-block
            c_cols: list of 2 col indices for C sub-block
            c_coeffs: dict mapping (dr, dc) to coefficient, where dr,dc in {0,1}
            Returns new col offset.
            """
            nonlocal col
            # 2x2 Strassen:
            # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
            # M2 = (a10+a11)b00       -> c10 += M2, c11 += M2
            # M3 = a00(b01-b11)       -> c01 += M3, c11 -= M3
            # M4 = a11(b10-b00)       -> c00 -= M4, c10 += M4
            # M5 = (a00+a01)b11       -> c00 += M5, c01 += M5
            # M6 = (a10-a00)(b00+b01) -> c11 -= M6
            # M7 = (a01-a11)(b10+b11) -> c00 -= M7
            
            # M1 = (a00+a11)(b00+b11)
            U[a_rows[0]*m + a_cols[0]][col] = 1
            U[a_rows[1]*m + a_cols[1]][col] = 1
            V[b_rows[0]*p + b_cols[0]][col] = 1
            V[b_rows[1]*p + b_cols[1]][col] = 1
            # c00 += M1, c11 += M1
            if (0,0) in c_coeffs:
                W[c_cols[0]*n + c_rows[0]][col] += c_coeffs[(0,0)]
            if (1,1) in c_coeffs:
                W[c_cols[1]*n + c_rows[1]][col] += c_coeffs[(1,1)]
            col += 1
            
            # M2 = (a10+a11)b00
            U[a_rows[1]*m + a_cols[0]][col] = 1
            U[a_rows[1]*m + a_cols[1]][col] = 1
            V[b_rows[0]*p + b_cols[0]][col] = 1
            if (1,0) in c_coeffs:
                W[c_cols[0]*n + c_rows[1]][col] += c_coeffs[(1,0)]
            if (1,1) in c_coeffs:
                W[c_cols[1]*n + c_rows[1]][col] += c_coeffs[(1,1)]
            col += 1
            
            # M3 = a00(b01-b11)
            U[a_rows[0]*m + a_cols[0]][col] = 1
            V[b_rows[0]*p + b_cols[1]][col] = 1
            V[b_rows[1]*p + b_cols[1]][col] = -1
            if (0,1) in c_coeffs:
                W[c_cols[1]*n + c_rows[0]][col] += c_coeffs[(0,1)]
            if (1,1) in c_coeffs:
                W[c_cols[1]*n + c_rows[1]][col] += c_coeffs[(1,1)] * (-1)
            col += 1
            
            # M4 = a11(b10-b00)
            U[a_rows[1]*m + a_cols[1]][col] = 1
            V[b_rows[1]*p + b_cols[0]][col] = 1
            V[b_rows[0]*p + b_cols[0]][col] = -1
            if (0,0) in c_coeffs:
                W[c_cols[0]*n + c_rows[0]][col] += c_coeffs[(0,0)] * (-1)
            if (1,0) in c_coeffs:
                W[c_cols[0]*n + c_rows[1]][col] += c_coeffs[(1,0)]
            col += 1
            
            # M5 = (a00+a01)b11
            U[a_rows[0]*m + a_cols[0]][col] = 1
            U[a_rows[0]*m + a_cols[1]][col] = 1
            V[b_rows[1]*p + b_cols[1]][col] = 1
            if (0,0) in c_coeffs:
                W[c_cols[0]*n + c_rows[0]][col] += c_coeffs[(0,0)]
            if (0,1) in c_coeffs:
                W[c_cols[1]*n + c_rows[0]][col] += c_coeffs[(0,1)]
            col += 1
            
            # M6 = (a10-a00)(b00+b01)
            U[a_rows[1]*m + a_cols[0]][col] = 1
            U[a_rows[0]*m + a_cols[0]][col] = -1
            V[b_rows[0]*p + b_cols[0]][col] = 1
            V[b_rows[0]*p + b_cols[1]][col] = 1
            if (1,1) in c_coeffs:
                W[c_cols[1]*n + c_rows[1]][col] += c_coeffs[(1,1)] * (-1)
            col += 1
            
            # M7 = (a01-a11)(b10+b11)
            U[a_rows[0]*m + a_cols[1]][col] = 1
            U[a_rows[1]*m + a_cols[1]][col] = -1
            V[b_rows[1]*p + b_cols[0]][col] = 1
            V[b_rows[1]*p + b_cols[1]][col] = 1
            if (0,0) in c_coeffs:
                W[c_cols[0]*n + c_rows[0]][col] += c_coeffs[(0,0)] * (-1)
            col += 1
            
            return col
        
        # Now define the 7 block products:
        # A00: rows 0-1, cols 0-1; A01: rows 0-1, cols 2-3
        # A10: rows 2-3, cols 0-1; A11: rows 2-3, cols 2-3
        # B00: rows 0-1, cols 0-1; B01: rows 0-1, cols 2-3
        # B10: rows 2-3, cols 0-1; B11: rows 2-3, cols 2-3
        # C00: rows 0-1, cols 0-1; C01: rows 0-1, cols 2-3
        # C10: rows 2-3, cols 0-1; C11: rows 2-3, cols 2-3
        
        # M1 = (A00+A11)(B00+B11)
        # A part: A00 + A11, so rows [0,2], cols [0,2] for the 2x2 "matrix"
        # But we need to handle the sums. The 2x2 product computes:
        # result[i,j] = sum over the 2x2 blocks
        # 
        # Actually, this approach is getting complicated. Let me use a different strategy.
        # 
        # I'll directly enumerate the 49 terms for 4x4 x 4x4 Strassen.
        # The 49 terms come from 7 block-products, each decomposed into 7 scalar products.
        
        # For M1 = (A00+A11)(B00+B11):
        # The "A" matrix has entries:
        #   A[0,0]+A[2,2], A[0,1]+A[2,3]
        #   A[1,0]+A[3,2], A[1,1]+A[3,3]
        # The "B" matrix has entries:
        #   B[0,0]+B[2,2], B[0,1]+B[2,3]
        #   B[1,0]+B[3,2], B[1,1]+B[3,3]
        # Result C00 = M1 (2x2 block), C11 += M1
        #
        # For the 2x2 Strassen on this "virtual" 2x2 matrix:
        # a00 = A[0,0]+A[2,2], a01 = A[0,1]+A[2,3]
        # a10 = A[1,0]+A[3,2], a11 = A[1,1]+A[3,3]
        # b00 = B[0,0]+B[2,2], b01 = B[0,1]+B[2,3]
        # b10 = B[1,0]+B[3,2], b11 = B[1,1]+B[3,3]
        #
        # M1_1 = (a00+a11)(b00+b11)
        # = (A[0,0]+A[2,2]+A[1,1]+A[3,3])(B[0,0]+B[2,2]+B[1,1]+B[3,3])
        # This is a product of sums, which expands to 16 terms, but in tensor decomposition
        # it's a single rank-1 term where U and V have multiple non-zero entries.
        
        # This is the key insight: each of the 49 terms is a rank-1 tensor where
        # U, V, W can have multiple non-zero entries.
        
        # Let me define a helper to add a term where:
        # U has specified (index, value) pairs
        # V has specified (index, value) pairs  
        # W has specified (index, value) pairs
        
        def add_term(U_entries, V_entries, W_entries, col_idx):
            for idx, val in U_entries:
                U[idx][col_idx] += val
            for idx, val in V_entries:
                V[idx][col_idx] += val
            for idx, val in W_entries:
                W[idx][col_idx] += val
        
        # For 4x4 x 4x4 Strassen, the 49 terms are:
        # Block product j (j=1..7), sub-product s (s=1..7)
        
        # Let me define the block-level Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each Mj, I need to compute a 2x2 x 2x2 product.
        # The 2x2 x 2x2 Strassen has 7 terms.
        
        # Let me define the A and B "virtual" 2x2 matrices for each Mj.
        
        # M1: A_virt[i,j] = A00[i,j] + A11[i,j], B_virt[i,j] = B00[i,j] + B11[i,j]
        # A_virt[0,0]=A[0,0]+A[2,2], A_virt[0,1]=A[0,1]+A[2,3]
        # A_virt[1,0]=A[1,0]+A[3,2], A_virt[1,1]=A[1,1]+A[3,3]
        # B_virt[0,0]=B[0,0]+B[2,2], B_virt[0,1]=B[0,1]+B[2,3]
        # B_virt[1,0]=B[1,0]+B[3,2], B_virt[1,1]=B[1,1]+B[3,3]
        # Result: C00 += M1_result, C11 += M1_result
        
        # For the 2x2 Strassen on M1:
        # s1: (a00+a11)(b00+b11) -> contributes to c00, c11
        # s2: (a10+a11)b00 -> contributes to c10, c11
        # s3: a00(b01-b11) -> contributes to c01, c11
        # s4: a11(b10-b00) -> contributes to c00, c10
        # s5: (a00+a01)b11 -> contributes to c00, c01
        # s6: (a10-a00)(b00+b01) -> contributes to c11
        # s7: (a01-a11)(b10+b11) -> contributes to c00
        
        # For s1: U gets a00 and a11, V gets b00 and b11, W gets c00 and c11
        # a00 = A[0,0]+A[2,2], so U entries: (0*m+0, 1), (2*m+2, 1)
        # a11 = A[1,1]+A[3,3], so U entries: (1*m+1, 1), (3*m+3, 1)
        # b00 = B[0,0]+B[2,2], so V entries: (0*p+0, 1), (2*p+2, 1)
        # b11 = B[1,1]+B[3,3], so V entries: (1*p+1, 1), (3*p+3, 1)
        # c00: C[0,0], C[0,1], C[1,0], C[1,1] (2x2 block at top-left)
        # c11: C[2,2], C[2,3], C[3,2], C[3,3] (2x2 block at bottom-right)
        # W for c00: (0*n+0, 1), (1*n+0, 1), (0*n+1, 1), (1*n+1, 1)
        # Wait, W[k*n+i] for C[i,k]. So C[0,0] -> W[0*4+0]=W[0]
        # C[0,1] -> W[1*4+0]=W[4]
        # C[1,0] -> W[0*4+1]=W[1]
        # C[1,1] -> W[1*4+1]=W[5]
        # C[2,2] -> W[2*4+2]=W[10]
        # C[2,3] -> W[3*4+2]=W[14]
        # C[3,2] -> W[2*4+3]=W[11]
        # C[3,3] -> W[3*4+3]=W[15]
        
        # This is very tedious but doable. Let me write a general function.
        
        # I'll define the 49 terms programmatically.
        
        # For each of the 7 block products, define:
        # - The 4 A-entries (as linear combos of A indices)
        # - The 4 B-entries (as linear combos of B indices)
        # - The C-output mapping (which 2x2 block of C gets the result, with sign)
        
        # Then for each, apply the 7-term 2x2 Strassen.
        
        # Let me define A_entries for each block product as a 2x2 list of (A_index, coeff) lists
        # Similarly for B_entries.
        
        # M1: A = A00+A11, B = B00+B11, C_out = [(C00, +1), (C11, +1)]
        # A_entries:
        #   [0,0]: [(0*m+0, 1), (2*m+2, 1)]
        #   [0,1]: [(0*m+1, 1), (2*m+3, 1)]
        #   [1,0]: [(1*m+0, 1), (3*m+2, 1)]
        #   [1,1]: [(1*m+1, 1), (3*m+3, 1)]
        # B_entries:
        #   [0,0]: [(0*p+0, 1), (2*p+2, 1)]
        #   [0,1]: [(0*p+1, 1), (2*p+3, 1)]
        #   [1,0]: [(1*p+0, 1), (3*p+2, 1)]
        #   [1,1]: [(1*p+1, 1), (3*p+3, 1)]
        
        # This is getting very complex. Let me just use schoolbook for the 4x4x4x4 part too,
        # giving rank 96, and then try to optimize.
        
        # Actually, let me just return schoolbook for now and ensure correctness.
        return col_offset
    
    # Given the complexity, let me just use pure schoolbook for correctness first.
    # Rank = 96
    
    rank = n * m * p  # 96
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
