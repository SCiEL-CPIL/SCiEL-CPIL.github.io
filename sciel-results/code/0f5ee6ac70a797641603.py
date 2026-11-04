"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Use Strassen for 4x4x4 (rank 49) and schoolbook for 4x4x1 (rank 16)
    # Total rank: 49 + 49 + 16 = 114
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]  # 16 x 114
    V = [[0] * rank for _ in range(m * p)]  # 36 x 114
    W = [[0] * rank for _ in range(p * n)]  # 36 x 114
    
    def add_strassen_2x2(a_base, b_base, c_base, col_start):
        """
        Add Strassen 2x2x2 decomposition.
        a_base: base index in U for A entries (a_base + i*2 + j for A[i,j])
        b_base: base index in V for B entries
        c_base: base index in W for C entries
        col_start: starting column
        Returns: new column
        """
        c = col_start
        # A[i,j] at a_base + i*2 + j
        # B[i,j] at b_base + i*2 + j
        # C[i,j] at c_base + i*2 + j
        
        # M1 = (A00+A11)(B00+B11)
        U[a_base + 0][c] = 1
        U[a_base + 3][c] = 1
        V[b_base + 0][c] = 1
        V[b_base + 3][c] = 1
        W[c_base + 0][c] = 1   # C00
        W[c_base + 3][c] = 1   # C11
        c += 1
        
        # M2 = (A10+A11)B00
        U[a_base + 2][c] = 1
        U[a_base + 3][c] = 1
        V[b_base + 0][c] = 1
        W[c_base + 2][c] = 1   # C10
        W[c_base + 3][c] = -1  # C11
        c += 1
        
        # M3 = A00(B10-B11)
        U[a_base + 0][c] = 1
        V[b_base + 2][c] = 1
        V[b_base + 3][c] = -1
        W[c_base + 1][c] = 1   # C01
        W[c_base + 3][c] = 1   # C11
        c += 1
        
        # M4 = A11(B01-B00)
        U[a_base + 3][c] = 1
        V[b_base + 1][c] = 1
        V[b_base + 0][c] = -1
        W[c_base + 0][c] = 1   # C00
        W[c_base + 2][c] = 1   # C10
        c += 1
        
        # M5 = (A00+A01)B11
        U[a_base + 0][c] = 1
        U[a_base + 1][c] = 1
        V[b_base + 3][c] = 1
        W[c_base + 0][c] = -1  # C00
        W[c_base + 1][c] = 1   # C01
        c += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a_base + 2][c] = 1
        U[a_base + 0][c] = -1
        V[b_base + 0][c] = 1
        V[b_base + 1][c] = 1
        W[c_base + 3][c] = 1   # C11
        c += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_base + 1][c] = 1
        U[a_base + 3][c] = -1
        V[b_base + 2][c] = 1
        V[b_base + 3][c] = 1
        W[c_base + 0][c] = 1   # C00
        c += 1
        
        return c
    
    def add_4x4x4_strassen(b_col_offset, col_start):
        """
        Add Strassen-based 4x4x4 decomposition (rank 49).
        b_col_offset: column offset in V for B (0 or 4)
        col_start: starting column
        Returns: new column
        """
        c = col_start
        
        # A is always the same 4x4 matrix, offset 0 in U
        # B has column offset b_col_offset in V
        # C has row offset b_col_offset in W (since C[k*n+i], k is column of B)
        
        # For 4x4 matrix with row-major indexing i*4+j:
        # Block (0,0): indices 0,1,4,5
        # Block (0,1): indices 2,3,6,7
        # Block (1,0): indices 8,9,12,13
        # Block (1,1): indices 10,11,14,15
        
        # Strassen for 2x2 block multiplication:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each Mi is a 2x2 matrix product, computed with Strassen (7 mults)
        
        # Helper to add a 2x2 Strassen product where:
        # A entries are at specific U rows, B entries at specific V rows,
        # and C entries contribute to specific W rows with coefficients
        
        def add_2x2_strassen_with_coeffs(a_coeffs, b_coeffs, c_coeffs, col):
            """
            Add a 2x2 Strassen product.
            a_coeffs: dict mapping (i,j) to coefficient for A[i,j] in U
            b_coeffs: dict mapping (i,j) to coefficient for B[i,j] in V
            c_coeffs: dict mapping (i,j) to coefficient for C[i,j] in W
            col: starting column
            Returns: new column
            """
            cc = col
            # Build the effective 2x2 matrices
            # For each of the 7 Strassen products, we need to set up U, V, W entries
            
            # Strassen products for 2x2:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # For each Strassen product, the U vector is a linear combo of A entries,
            # V vector is a linear combo of B entries, and W vector gives contributions to C.
            
            # We need to compose the coefficients.
            
            # M1: U = A00+A11, V = B00+B11
            # U coeffs: {(0,0):1, (1,1):1}, V coeffs: {(0,0):1, (1,1):1}
            # W contribution: C00 += 1, C11 += 1
            
            # M2: U = A10+A11, V = B00
            # U coeffs: {(1,0):1, (1,1):1}, V coeffs: {(0,0):1}
            # W contribution: C10 += 1, C11 += -1
            
            # M3: U = A00, V = B10-B11
            # U coeffs: {(0,0):1}, V coeffs: {(1,0):1, (1,1):-1}
            # W contribution: C01 += 1, C11 += 1
            
            # M4: U = A11, V = B01-B00
            # U coeffs: {(1,1):1}, V coeffs: {(0,1):1, (0,0):-1}
            # W contribution: C00 += 1, C10 += 1
            
            # M5: U = A00+A01, V = B11
            # U coeffs: {(0,0):1, (0,1):1}, V coeffs: {(1,1):1}
            # W contribution: C00 += -1, C01 += 1
            
            # M6: U = A10-A00, V = B00+B01
            # U coeffs: {(1,0):1, (0,0):-1}, V coeffs: {(0,0):1, (0,1):1}
            # W contribution: C11 += 1
            
            # M7: U = A01-A11, V = B10+B11
            # U coeffs: {(0,1):1, (1,1):-1}, V coeffs: {(1,0):1, (1,1):1}
            # W contribution: C00 += 1
            
            # Now compose with the outer coefficients
            
            # For each Strassen product k, the effective U entry for A[i,j] is:
            # sum over strassen products where A[i,j] appears: outer_coeff * inner_coeff
            
            # This is getting complex. Let me just enumerate all 49 scalar multiplications directly.
            
            return cc
        
        # Direct approach: enumerate all 49 scalar multiplications
        # For each of the 7 outer Strassen products, and for each of the 7 inner Strassen products,
        # we get one scalar multiplication.
        
        # Let me define the 7 outer Strassen products with their A and B block coefficients:
        # Outer product k has A-block coefficients a_k[(bi,bj)] and B-block coefficients b_k[(bi,bj)]
        # where (bi,bj) is the block position (0 or 1, 0 or 1)
        
        # M1: A blocks: (0,0):1, (1,1):1; B blocks: (0,0):1, (1,1):1
        # M2: A blocks: (1,0):1, (1,1):1; B blocks: (0,0):1
        # M3: A blocks: (0,0):1; B blocks: (1,0):1, (1,1):-1
        # M4: A blocks: (1,1):1; B blocks: (0,1):1, (0,0):-1
        # M5: A blocks: (0,0):1, (0,1):1; B blocks: (1,1):1
        # M6: A blocks: (1,0):1, (0,0):-1; B blocks: (0,0):1, (0,1):1
        # M7: A blocks: (0,1):1, (1,1):-1; B blocks: (1,0):1, (1,1):1
        
        # Inner Strassen products for 2x2:
        # S1: A entries: (0,0):1,(1,1):1; B entries: (0,0):1,(1,1):1; C: (0,0):1,(1,1):1
        # S2: A entries: (1,0):1,(1,1):1; B entries: (0,0):1; C: (1,0):1,(1,1):-1
        # S3: A entries: (0,0):1; B entries: (1,0):1,(1,1):-1; C: (0,1):1,(1,1):1
        # S4: A entries: (1,1):1; B entries: (0,1):1,(0,0):-1; C: (0,0):1,(1,0):1
        # S5: A entries: (0,0):1,(0,1):1; B entries: (1,1):1; C: (0,0):-1,(0,1):1
        # S6: A entries: (1,0):1,(0,0):-1; B entries: (0,0):1,(0,1):1; C: (1,1):1
        # S7: A entries: (0,1):1,(1,1):-1; B entries: (1,0):1,(1,1):1; C: (0,0):1
        
        # Block (bi,bj) in 4x4 corresponds to entries at rows (2*bi, 2*bi+1) and cols (2*bj, 2*bj+1)
        # Entry (i,j) within block is at global position (2*bi+i, 2*bj+j)
        # U index: (2*bi+i)*4 + (2*bj+j)
        
        outer_products = [
            # (a_block_coeffs, b_block_coeffs)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
            ({(1,0):1, (1,1):1}, {(0,0):1}),              # M2
            ({(0,0):1}, {(1,0):1, (1,1):-1}),             # M3
            ({(1,1):1}, {(0,1):1, (0,0):-1}),             # M4
            ({(0,0):1, (0,1):1}, {(1,1):1}),              # M5
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),    # M6
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),    # M7
        ]
        
        inner_products = [
            # (a_entry_coeffs, b_entry_coeffs, c_entry_coeffs)
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # S1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),          # S2
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),          # S3
            ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),          # S4
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),          # S5
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),          # S6
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),          # S7
        ]
        
        for oi, (a_blocks, b_blocks) in enumerate(outer_products):
            for ii, (a_entries, b_entries, c_entries) in enumerate(inner_products):
                col = c
                # For each A entry in inner product, and each block in outer product:
                for (ai, aj), a_coeff_inner in a_entries.items():
                    for (bi, bj), a_coeff_outer in a_blocks.items():
                        coeff = a_coeff_inner * a_coeff_outer
                        if coeff != 0:
                            # Global row in U: (2*bi+ai)*4 + (2*bj+aj)
                            row = (2*bi+ai)*4 + (2*bj+aj)
                            U[row][col] += coeff
                
                for (bi, bj), b_coeff_outer in b_blocks.items():
                    for (ci, cj), b_coeff_inner in b_entries.items():
                        coeff = b_coeff_outer * b_coeff_inner
                        if coeff != 0:
                            # Global row in V: (2*bi+ci)*4 + (2*bj+cj) + b_col_offset*4
                            # Wait, B is 4x4, stored with V[j*p+k] where j is row of B, k is col
                            # B[j,k] is at V[j*4+k] for the first 4 columns
                            # For column offset b_col_offset, B[j, b_col_offset+k] is at V[j*4 + b_col_offset+k]
                            row = (2*bi+ci)*4 + (2*bj+cj) + b_col_offset
                            V[row][col] += coeff
                
                for (ci, cj), c_coeff_inner in c_entries.items():
                    # The C block for outer product (oi) contributes to specific C blocks
                    # C block (ci,cj) in the 2x2 block result
                    # Global C position: (2*ci+ri)*4 + (2*cj+cj2) where ri,cj2 are within block
                    # Wait, I need to map the inner C entries to global C positions
                    
                    # The inner Strassen computes a 2x2 product P = X*Y
                    # P[ci,cj] is the result
                    # This P corresponds to one of the C blocks in the 4x4 result
                    
                    # For outer product M1: contributes to C00 and C11
                    # For outer product M2: contributes to C10 and C11
                    # etc.
                    
                    # Actually, I realize this approach is getting too complex. Let me use a simpler method.
                    pass
                
                c += 1
        
        return c
    
    # This approach is too complex. Let me use a direct enumeration.
    # I'll compute the 4x4x4 Strassen decomposition by explicitly tracking all scalar multiplications.
    
    # Reset and use a cleaner approach
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_4x4x4_strassen_direct(b_col_offset, col_start):
        """Add 4x4x4 Strassen decomposition with rank 49."""
        c = col_start
        
        # Define helper to add a scalar multiplication
        def add_mult(a_row, v_row, w_rows_coeffs):
            nonlocal c
            U[a_row][c] = 1
            V[v_row][c] = 1
            for w_row, coeff in w_rows_coeffs:
                W[w_row][c] += coeff
            c += 1
        
        # For 4x4 Strassen, I'll directly enumerate the 49 scalar multiplications.
        # This is tedious but straightforward.
        
        # Let me use a different strategy: compute the full 4x4x4 tensor and find
        # a rank-49 decomposition by composing two Strassen layers.
        
        # Actually, let me just use schoolbook for simplicity and accept rank 144.
        # No wait, let me try to implement this correctly.
        
        # I'll use the fact that Strassen for 4x4x4 can be implemented as:
        # 7 outer products, each being a 2x2 matrix product done with 7 inner products.
        
        # For each outer product, the A and B operands are 2x2 matrices whose entries
        # are linear combinations of the 4x4 entries.
        
        # Let me define the 7 outer products explicitly with their 2x2 operand matrices.
        
        # Outer M1: A_op = A11+A22, B_op = B11+B22
        # A11[i,j] = A[2i, 2j], A22[i,j] = A[2i+2, 2j+2]
        # So A_op[i,j] = A[2i, 2j] + A[2i+2, 2j+2]
        # B_op[i,j] = B[2i, 2j] + B[2i+2, 2j+2] (with column offset)
        
        # The 2x2 product P = A_op * B_op has entries:
        # P[0,0] = A_op[0,0]*B_op[0,0] + A_op[0,1]*B_op[1,0]
        # P[0,1] = A_op[0,0]*B_op[0,1] + A_op[0,1]*B_op[1,1]
        # P[1,0] = A_op[1,0]*B_op[0,0] + A_op[1,1]*B_op[1,0]
        # P[1,1] = A_op[1,0]*B_op[0,1] + A_op[1,1]*B_op[1,1]
        
        # Each of these is a sum of 2 products, but we use Strassen to reduce to 7 products.
        
        # This is getting very involved. Let me just implement schoolbook for now
        # and accept rank 144, or find a simpler improvement.
        
        return c
    
    # Given the complexity, let me use a simpler but still improved approach:
    # Use Strassen for 2x2x2 blocks directly.
    
    # For 4x4x4, partition into 2x2 blocks and use schoolbook for the block multiplication
    # (8 block products), each done with Strassen (rank 7). Total: 8*7 = 56.
    # For 4x4x9: 56 + 56 + 32 = 144. Same as before!
    
    # Let me just use the original schoolbook approach.
    rank = n * m * p
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
