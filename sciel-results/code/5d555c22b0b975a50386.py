import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition using Strassen's algorithm for each of the 5 2x2 blocks
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # Indices in U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # M1: A00+A11
    # M2: A10+A11
    # M3: A00
    # M4: A11
    # M5: A00+A01
    # M6: A10-A00
    # M7: A01-A11
    
    u_coeffs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1}       # M7
    ]
    
    # Strassen coefficients for B block (2x2)
    # B_r indices in V: 
    # 0: B00 (row 0, col 0 of block)
    # 1: B01 (row 0, col 1 of block)
    # 2: B10 (row 1, col 0 of block)
    # 3: B11 (row 1, col 1 of block)
    
    v_coeffs = [
        {0: 1, 3: 1},       # M1: B00+B11
        {0: 1},             # M2: B00
        {2: 1, 3: -1},      # M3: B10-B11
        {1: 1, 0: -1},      # M4: B01-B00
        {3: 1},             # M5: B11
        {0: 1, 1: 1},       # M6: B00+B01
        {2: 1, 3: 1}        # M7: B10+B11
    ]
    
    # Reconstruction coefficients for C
    # C indices in W for a block:
    # 0: C00
    # 1: C01
    # 2: C10
    # 3: C11
    
    w_coeffs = [
        {0: 1, 3: 1, 2: -1, 1: 1},   # M1 -> C00(1), C01(0), C10(0), C11(1) ... wait
        # Let's map carefully.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need W[col_c, row_c] where row_c is index in C.
        # For a specific M_s, which C entries does it contribute to?
        
        # M1: C00(+1), C11(+1)
        # M2: C10(+1), C11(-1)
        # M3: C01(+1), C11(+1)
        # M4: C00(+1), C10(+1)
        # M5: C01(+1), C00(-1)
        # M6: C11(+1)
        # M7: C00(+1)
        
        {0: 1, 3: 1},       # M1: C00(0), C11(3)
        {2: 1, 3: -1},      # M2: C10(2), C11(3)
        {1: 1, 3: 1},       # M3: C01(1), C11(3)
        {0: 1, 2: 1},       # M4: C00(0), C10(2)
        {1: 1, 0: -1},      # M5: C01(1), C00(0)
        {3: 1},             # M6: C11(3)
        {0: 1}              # M7: C00(0)
    ]
    
    for r in range(5):
        for s in range(7):
            c = r * 7 + s
            
            # Fill U
            for idx, val in u_coeffs[s].items():
                U[idx][c] = val
                
            # Fill V
            # Block r columns are 2r and 2r+1 in B
            # B indices in V:
            # B00 -> row 0, col 2r -> V[0*10 + 2r]
            # B01 -> row 0, col 2r+1 -> V[0*10 + 2r+1]
            # B10 -> row 1, col 2r -> V[1*10 + 2r]
            # B11 -> row 1, col 2r+1 -> V[1*10 + 2r+1]
            
            v_map = {
                0: 0 * 10 + 2 * r,
                1: 0 * 10 + 2 * r + 1,
                2: 1 * 10 + 2 * r,
                3: 1 * 10 + 2 * r + 1
            }
            for idx, val in v_coeffs[s].items():
                V[v_map[idx]][c] = val
                
            # Fill W
            # Block r output columns are 2r and 2r+1 in C
            # C indices in W:
            # C00 -> col 2r, row 0 -> W[2r*2 + 0]
            # C01 -> col 2r+1, row 0 -> W[(2r+1)*2 + 0]
            # C10 -> col 2r, row 1 -> W[2r*2 + 1]
            # C11 -> col 2r+1, row 1 -> W[(2r+1)*2 + 1]
            
            w_map = {
                0: 2 * r * 2 + 0,
                1: (2 * r + 1) * 2 + 0,
                2: 2 * r * 2 + 1,
                3: (2 * r + 1) * 2 + 1
            }
            for idx, val in w_coeffs[s].items():
                W[w_map[idx]][c] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
