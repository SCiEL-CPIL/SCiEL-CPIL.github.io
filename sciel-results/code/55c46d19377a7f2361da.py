"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We are using a block decomposition with 2x2 blocks of 3x3 matrices.
    # We use Strassen's algorithm for the 2x2 block multiplication.
    # Each of the 7 Strassen multiplications is a 3x3 matrix multiplication.
    # We expand these 3x3 multiplications into scalar rank-1 terms.
    
    # Total rank calculation:
    # M1: (A11+A22)(B11+B22) -> 2*2 = 4 terms per 3x3 entry, 9 entries -> 36
    # M2: (A21+A22)B11 -> 2*1 = 2 terms per 3x3 entry, 9 entries -> 18
    # M3: A11(B12-B22) -> 1*2 = 2 terms per 3x3 entry, 9 entries -> 18
    # M4: A22(B21-B11) -> 1*2 = 2 terms per 3x3 entry, 9 entries -> 18
    # M5: (A11+A12)B22 -> 2*1 = 2 terms per 3x3 entry, 9 entries -> 18
    # M6: (A21-A11)(B11+B12) -> 2*2 = 4 terms per 3x3 entry, 9 entries -> 36
    # M7: (A12-A22)(B21+B22) -> 2*2 = 4 terms per 3x3 entry, 9 entries -> 36
    # Total = 36+18+18+18+18+36+36 = 180
    
    rank = 180
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to get global index for a block entry
    # Block (br, bc) of size 3x3
    # Entry (i, j) in the block
    def block_idx(br, bc, i, j):
        return (br * 3 + i) * 6 + (bc * 3 + j)

    # Define the 7 Strassen steps
    # Each step defines:
    # X_coeffs: list of (block_row, block_col, coeff) for A
    # Y_coeffs: list of (block_row, block_col, coeff) for B
    # C_coeffs: list of (block_row, block_col, coeff) for C contribution
    
    # M1 = (A11 + A22)(B11 + B22)
    # X: A11(1), A22(1)
    # Y: B11(1), B22(1)
    # C: C11(1)
    steps = [
        {
            'X': [((0,0), 1), ((1,1), 1)],
            'Y': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        },
        {
            'X': [((1,0), 1), ((1,1), 1)],
            'Y': [((0,0), 1)],
            'C': [((1,0), 1)]
        },
        {
            'X': [((0,0), 1)],
            'Y': [((0,1), 1), ((1,1), -1)],
            'C': [((0,1), 1)]
        },
        {
            'X': [((1,1), 1)],
            'Y': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1)]
        },
        {
            'X': [((0,0), 1), ((0,1), 1)],
            'Y': [((1,1), 1)],
            'C': [((0,0), -1)]
        },
        {
            'X': [((1,0), 1), ((0,0), -1)],
            'Y': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        {
            'X': [((0,1), 1), ((1,1), -1)],
            'Y': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        }
    ]
    
    # Wait, I need to check the C coefficients for Strassen.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's re-verify the C contributions for each M:
    # M1: C11(1), C22(1)
    # M2: C21(1), C22(-1)
    # M3: C12(1), C22(1)
    # M4: C11(1), C21(1)
    # M5: C11(-1), C12(1)
    # M6: C22(1)
    # M7: C11(1)
    
    # My previous 'C' list in the steps dict was incomplete. I need to apply the full Strassen combination.
    # It is easier to define the X and Y for each M, and then for each scalar term, 
    # determine which C blocks it contributes to and with what sign.
    
    # Revised Steps Definition:
    # X: list of (br, bc, coeff)
    # Y: list of (br, bc, coeff)
    # C_contrib: list of (br, bc, coeff)
    
    steps = [
        { # M1
            'X': [((0,0), 1), ((1,1), 1)],
            'Y': [((0,0), 1), ((1,1), 1)],
            'C': [((0,0), 1), ((1,1), 1)]
        },
        { # M2
            'X': [((1,0), 1), ((1,1), 1)],
            'Y': [((0,0), 1)],
            'C': [((1,0), 1), ((1,1), -1)]
        },
        { # M3
            'X': [((0,0), 1)],
            'Y': [((0,1), 1), ((1,1), -1)],
            'C': [((0,1), 1), ((1,1), 1)]
        },
        { # M4
            'X': [((1,1), 1)],
            'Y': [((1,0), 1), ((0,0), -1)],
            'C': [((0,0), 1), ((1,0), 1)]
        },
        { # M5
            'X': [((0,0), 1), ((0,1), 1)],
            'Y': [((1,1), 1)],
            'C': [((0,0), -1), ((0,1), 1)]
        },
        { # M6
            'X': [((1,0), 1), ((0,0), -1)],
            'Y': [((0,0), 1), ((0,1), 1)],
            'C': [((1,1), 1)]
        },
        { # M7
            'X': [((0,1), 1), ((1,1), -1)],
            'Y': [((1,0), 1), ((1,1), 1)],
            'C': [((0,0), 1)]
        }
    ]
    
    for step in steps:
        X_blocks = step['X']
        Y_blocks = step['Y']
        C_blocks = step['C']
        
        # Iterate over the 9 terms of the 3x3 matrix multiplication
        for i in range(3):
            for l in range(3):
                for j in range(3):
                    # The term is X[i,l] * Y[l,j]
                    # X[i,l] is sum of c_x * A_block[br,bc][i,l]
                    # Y[l,j] is sum of c_y * B_block[br,bc][l,j]
                    
                    # Expand the product
                    for (bx, cx) in X_blocks:
                        for (by, cy) in Y_blocks:
                            coeff = cx * cy
                            
                            # A entry: block (bx, cx) is wrong notation.
                            # bx, by are block indices.
                            # Let's rename: (brx, bcx, cx) and (bry, bcy, cy)
                            
                            pass # Placeholder, will rewrite loop below

    # Rewriting the loop properly
    col = 0
    for step in steps:
        X_blocks = step['X'] # List of (br, bc, coeff)
        Y_blocks = step['Y']
        C_blocks = step['C']
        
        for i in range(3):
            for l in range(3):
                for j in range(3):
                    # Iterate over all pairs of blocks in X and Y
                    for (brx, bcx, cx) in X_blocks:
                        for (bry, bcy, cy) in Y_blocks:
                            coeff = cx * cy
                            
                            # A index: block (brx, bcx), entry (i, l)
                            a_idx = block_idx(brx, bcx, i, l)
                            # B index: block (bry, bcy), entry (l, j)
                            b_idx = block_idx(bry, bcy, l, j)
                            
                            # C contribution
                            # This term contributes to M_step[i,j]
                            # M_step[i,j] contributes to C blocks defined in C_blocks
                            # The entry in C is at (i,j) within the block
                            
                            # We need to add this rank-1 term to U, V, W
                            # U[a_idx, col] += coeff
                            # V[b_idx, col] += coeff
                            # W[c_idx, col] += coeff * c_coeff
                            
                            # Wait, the rank-1 term is A_u * B_v * C_w.
                            # The coefficient is cx * cy * c_coeff.
                            
                            for (brc, bcc, cc) in C_blocks:
                                c_idx = block_idx(brc, bcc, i, j)
                                total_coeff = cx * cy * cc
                                
                                U[a_idx][col] += total_coeff
                                V[b_idx][col] += total_coeff
                                W[c_idx][col] += total_coeff
                            
                            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
