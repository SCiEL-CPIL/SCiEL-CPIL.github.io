"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm on 2x2 blocks of 3x3 matrices.
    # Rank = 7 * 27 = 189.
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 7 Strassen products.
    # Each product is defined by:
    # - X blocks: list of (block_id, coeff)
    # - Y blocks: list of (block_id, coeff)
    # - Destinations: list of (dest_block_id, coeff)
    
    # Block IDs: 00, 01, 10, 11
    # 00: rows 0-2, cols 0-2
    # 01: rows 0-2, cols 3-5
    # 10: rows 3-5, cols 0-2
    # 11: rows 3-5, cols 3-5
    
    strassen_products = [
        {
            "X": [(0, 1), (3, 1)],  # A00 + A11
            "Y": [(0, 1), (3, 1)],  # B00 + B11
            "Dests": [(0, 1), (3, 1)] # C00 + C11
        },
        {
            "X": [(2, 1), (3, 1)],  # A10 + A11
            "Y": [(0, 1)],          # B00
            "Dests": [(2, 1), (3, -1)] # C10 - C11
        },
        {
            "X": [(0, 1)],          # A00
            "Y": [(1, 1), (3, -1)], # B01 - B11
            "Dests": [(1, 1), (3, 1)] # C01 + C11
        },
        {
            "X": [(3, 1)],          # A11
            "Y": [(2, 1), (0, -1)], # B10 - B00
            "Dests": [(0, 1), (2, 1)] # C00 + C10
        },
        {
            "X": [(0, 1), (1, 1)],  # A00 + A01
            "Y": [(3, 1)],          # B11
            "Dests": [(0, -1), (1, 1)] # -C00 + C01
        },
        {
            "X": [(2, 1), (0, -1)], # A10 - A00
            "Y": [(0, 1), (1, 1)],  # B00 + B01
            "Dests": [(3, 1)]       # C11
        },
        {
            "X": [(1, 1), (3, -1)], # A01 - A11
            "Y": [(2, 1), (3, 1)],  # B10 + B11
            "Dests": [(0, 1)]       # C00
        }
    ]
    
    for prod in strassen_products:
        X_blocks = prod["X"]
        Y_blocks = prod["Y"]
        Dests = prod["Dests"]
        
        # Iterate over local indices i, j, l for the 3x3 blocks
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    # For each combination of X block and Y block
                    for bx, cx in X_blocks:
                        for by, cy in Y_blocks:
                            # Determine global indices for A and B
                            # bx: 0->00, 1->01, 2->10, 3->11
                            # i, l are local indices 0..2
                            
                            # A block bx
                            # Row: i + (3 if bx in [2,3] else 0)
                            # Col: l + (3 if bx in [1,3] else 0)
                            row_a = i + (3 if bx >= 2 else 0)
                            col_a = l + (3 if bx % 2 == 1 else 0)
                            
                            # B block by
                            # Row: l + (3 if by in [2,3] else 0)
                            # Col: j + (3 if by in [1,3] else 0)
                            row_b = l + (3 if by >= 2 else 0)
                            col_b = j + (3 if by % 2 == 1 else 0)
                            
                            # Check if col_a == row_b (should always be true for valid Strassen)
                            if col_a != row_b:
                                continue
                            
                            # This term A[row_a, col_a] * B[row_b, col_b] contributes to M_g[i,j]
                            # M_g[i,j] contributes to Dests
                            
                            for bd, cd in Dests:
                                # Determine global indices for C
                                # C block bd
                                # Row: i + (3 if bd in [2,3] else 0)
                                # Col: j + (3 if bd in [1,3] else 0)
                                row_c = i + (3 if bd >= 2 else 0)
                                col_c = j + (3 if bd % 2 == 1 else 0)
                                
                                # Update U, V, W
                                # U index: row_a * m + col_a
                                # V index: row_b * p + col_b
                                # W index: col_c * n + row_c (Trace convention)
                                
                                U[row_a * m + col_a][col] = cx
                                V[row_b * p + col_b][col] = cy
                                W[col_c * n + row_c][col] = cd
                                
                                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
