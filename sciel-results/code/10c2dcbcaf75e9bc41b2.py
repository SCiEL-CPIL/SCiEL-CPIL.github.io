"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm for each of the 3 blocks of size 2x2 in the p dimension.
    # p=6, so there are 3 blocks.
    # Each 2x2x2 multiplication has rank 7.
    # Total rank = 3 * 7 = 21.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)] # 4 x 21
    V = [[0] * rank for _ in range(m * p)] # 12 x 21
    W = [[0] * rank for _ in range(p * n)] # 12 x 21
    
    # Strassen components for 2x2 multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # U vectors (coefficients for A)
    # Index in A: i*m + j. m=2.
    # 0: a00, 1: a01, 2: a10, 3: a11
    U_forms = [
        [1, 0, 0, 1],  # M1: a00 + a11
        [0, 0, 1, 1],  # M2: a10 + a11
        [1, 0, 0, 0],  # M3: a00
        [0, 0, 0, 1],  # M4: a11
        [1, 1, 0, 0],  # M5: a00 + a01
        [-1, 0, 1, 0], # M6: a10 - a00
        [0, 1, 0, -1]  # M7: a01 - a11
    ]
    
    # V vectors (coefficients for B) for a single block
    # B is 2x6. Block r uses columns 2r and 2r+1.
    # Local indices in block:
    # b00 is at col 2r, row 0. Global index: 0*6 + 2r
    # b01 is at col 2r+1, row 0. Global index: 0*6 + 2r+1
    # b10 is at col 2r, row 1. Global index: 1*6 + 2r
    # b11 is at col 2r+1, row 1. Global index: 1*6 + 2r+1
    
    # We define the local pattern for V for each M_s.
    # Pattern is a dict mapping local (row, col_in_block) to coefficient.
    # col_in_block is 0 or 1.
    V_patterns = [
        {(0,0): 1, (1,1): 1},       # M1: b00 + b11
        {(0,0): 1},                 # M2: b00
        {(0,1): 1, (1,1): -1},      # M3: b01 - b11
        {(1,0): 1, (0,0): -1},      # M4: b10 - b00
        {(1,1): 1},                 # M5: b11
        {(0,0): 1, (0,1): 1},       # M6: b00 + b01
        {(1,0): 1, (1,1): 1}        # M7: b10 + b11
    ]
    
    # W vectors (coefficients for C) for a single block
    # C is 2x6. Block r uses columns 2r and 2r+1.
    # Global index: i*6 + (2r + col_in_block).
    # W_s has non-zero entries at the 4 positions of C for block r.
    # Coefficients are from Strassen reconstruction:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Map s (0..6) to coefficients for c00, c01, c10, c11
    # Order of outputs in W for block r:
    # idx_c00 = 0*6 + 2r
    # idx_c01 = 0*6 + 2r+1
    # idx_c10 = 1*6 + 2r
    # idx_c11 = 1*6 + 2r+1
    
    W_coeffs = [
        [1, 0, 0, 1],  # M1: c00+1, c01+0, c10+0, c11+1
        [0, 0, 1, -1], # M2: c00+0, c01+0, c10+1, c11-1
        [0, 1, 0, 1],  # M3: c00+0, c01+1, c10+0, c11+1
        [1, 0, 1, 0],  # M4: c00+1, c01+0, c10+1, c11+0
        [-1, 1, 0, 0], # M5: c00-1, c01+1, c10+0, c11+0
        [0, 0, 0, 1],  # M6: c00+0, c01+0, c10+0, c11+1
        [1, 0, 0, 0]   # M7: c00+1, c01+0, c10+0, c11+0
    ]
    
    col = 0
    for r in range(3): # 3 blocks
        for s in range(7): # 7 Strassen steps
            # Fill U
            for i in range(4):
                U[i][col] = U_forms[s][i]
            
            # Fill V
            # V is 12x21.
            # For this block r, only rows 0 and 1 have non-zeros, at columns 2r and 2r+1.
            for (row, col_in_block), val in V_patterns[s].items():
                global_row = row
                global_col = 2 * r + col_in_block
                # V index is row * p + col. p=6.
                V[global_row * 6 + global_col][col] = val
            
            # Fill W
            # W is 12x21.
            # Non-zeros at:
            # c00: row 0, col 2r
            # c01: row 0, col 2r+1
            # c10: row 1, col 2r
            # c11: row 1, col 2r+1
            coeffs = W_coeffs[s]
            W[0 * 6 + 2 * r][col] = coeffs[0]
            W[0 * 6 + 2 * r + 1][col] = coeffs[1]
            W[1 * 6 + 2 * r][col] = coeffs[2]
            W[1 * 6 + 2 * r + 1][col] = coeffs[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
