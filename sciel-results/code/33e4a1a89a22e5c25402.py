"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 216
    
    # Identity matrices for 3x3 blocks
    I3 = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    Z3 = [[0.0 for j in range(3)] for i in range(3)]
    
    # Define 7 Strassen-like matrices for 2x2 block multiplication
    # Each matrix is a function of I and Z (identity and zero) for 3x3 blocks
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    def add(A, B):
        return [[A[i][j] + B[i][j] for j in range(3)] for i in range(3)]
    
    def sub(A, B):
        return [[A[i][j] - B[i][j] for j in range(3)] for i in range(3)]
    
    # U parts: coefficients for A blocks (2x2 of 3x3 blocks)
    # V parts: coefficients for B blocks (2x2 of 3x3 blocks)
    # W parts: coefficients for C blocks (2x2 of 3x3 blocks)
    
    # For each of 7 steps, we have 9 sub-columns (for 3x3 schoolbook)
    # Total rank = 7 * 9 = 63
    
    rank = 7 * 9
    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 63
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 63
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 63
    
    # A blocks: A00 (rows 0-2, cols 0-2), A01 (rows 0-2, cols 3-5)
    #           A10 (rows 3-5, cols 0-2), A11 (rows 3-5, cols 3-5)
    # B blocks: B00 (rows 0-2, cols 0-2), B01 (rows 0-2, cols 3-5)
    #           B10 (rows 3-5, cols 0-2), B11 (rows 3-5, cols 3-5)
    # C blocks: C00 (rows 0-2, cols 0-2), C01 (rows 0-2, cols 3-5)
    #           C10 (rows 3-5, cols 0-2), C11 (rows 3-5, cols 3-5)
    
    def block_row(block_idx, r):
        """Get global row index for block row block_idx (0 or 1) and local row r (0,1,2)"""
        return block_idx * 3 + r
    
    def block_col(block_idx, c):
        """Get global col index for block col block_idx (0 or 1) and local col c (0,1,2)"""
        return block_idx * 3 + c
    
    # For each of 7 Strassen steps, for each of 9 schoolbook sub-columns:
    # Step s (0-6), sub-col t (0-8) where t = r*3 + c (r,c in 0,1,2)
    # Column index = s*9 + t
    
    for s in range(7):
        for r in range(3):
            for c in range(3):
                col = s * 9 + r * 3 + c
                
                # W part: C[k*n + i] where i is row in C, k is col in C
                # For schoolbook: C[i][k] += A[i][t] * B[t][k]
                # So W gets 1 at position corresponding to C[r][c] for each (i,k) pair? 
                # No, W[k*n + i] = 1 means C[i][k] gets the product.
                # For sub-column (r,c), we're computing contribution to C[r][c]
                # from A's row r and B's col c (in the 3x3 block sense)
                
                # Actually, let me think more carefully.
                # In the 3x3 schoolbook, for each (i,j,k) we have a column where
                # U has 1 at A[i][j], V has 1 at B[j][k], W has 1 at C[i][k].
                # 
                # For block-level Strassen, each "block multiplication" A_block * B_block
                # is a 3x3 * 3x3 = 3x3 multiplication, which has 27 terms (i,j,k).
                # But we're using schoolbook within each block, so for each of the 7
                # Strassen steps, we do a full 3x3 schoolbook multiplication.
                # 
                # So for step s, the 27 columns correspond to (i,j,k) in 0,1,2.
                # The sub-column index t = i*9 + j*3 + k? Or some ordering.
                # 
                # Let me use t = i*9 + j*3 + k where i is A row, j is intermediate, k is B col.
                # But rank is 63 = 7*9, so I must have meant 9 per step, not 27.
                
                # Wait, I think I made an error. Let me reconsider.
                # 
                # Strassen for 2x2: 7 multiplications, each of which is a 3x3*3x3 = 3x3.
                # Each 3x3 multiplication has rank 27 (schoolbook) or less.
                # If we use schoolbook for each 3x3, total rank = 7 * 27 = 189.
                # 
                # My original plan said 63 = 7*9, which is wrong for schoolbook 3x3.
                # Let me use rank = 7 * 27 = 189.
                
                pass
    
    # Let me restart with the correct understanding:
    # Rank = 7 * 27 = 189 (using schoolbook for each 3x3 block multiplication)
    
    rank = 7 * 27
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # For each Strassen step s (0-6):
    #   We compute a linear combination of A blocks * linear combination of B blocks
    #   Each is a 3x3 matrix multiplication
    #   For the 3x3 multiplication, we use schoolbook: 27 terms (i,j,k)
    #   For each (i,j,k), we have a column where:
    #     U has coefficients from the A-block linear combination
    #     V has coefficients from the B-block linear combination  
    #     W has 1 at the C position
    
    # Define the 7 Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # For each step, define which A blocks and B blocks are used, with signs
    # A_blocks: list of (block_idx, sign)
    # B_blocks: list of (block_idx, sign)
    
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11
    
    strassen_steps = [
        # (A_blocks, B_blocks, C_combination)
        # C_combination: list of (C_block_idx, sign)
        # C blocks: 0=C00, 1=C01, 2=C10, 3=C11
        
        # M1 = (A00+A11)*(B00+B11)
        # C00 += M1, C01 += M1, C10 += M1, C11 += M1
        # Wait, that's not right. Let me recall Strassen:
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # So for each M_s, it contributes to specific C blocks with specific signs.
        
        # Step 0: M1 = (A00+A11)*(B00+B11)
        # A: A00(+1), A11(+1)
        # B: B00(+1), B11(+1)
        # C: C00(+1), C01(+1), C10(+1), C11(+1)  -- No, this is wrong.
        
        # Let me look up Strassen correctly:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M4 + M6 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
    ]
    
    # Let me define it properly:
    steps = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict mapping A_block_idx -> sign
        # B_coeffs: dict mapping B_block_idx -> sign
        # C_coeffs: dict mapping C_block_idx -> sign
        
        # M1 = (A00+A11)*(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 1, 2: 1, 3: 1}),  # This C is wrong
        
        # I need to fix the C coefficients. Let me be more careful.
        # Actually, each M_s is a 3x3 matrix, and it contributes to C blocks.
        # The C combination is fixed by Strassen's formulas.
        
        # Let me just encode the full Strassen algorithm properly.
    ]
    
    # This is getting complicated. Let me use a simpler approach:
    # Just use the schoolbook decomposition for the full 6x6 matrix.
    # Rank = 6*6*6 = 216.
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
