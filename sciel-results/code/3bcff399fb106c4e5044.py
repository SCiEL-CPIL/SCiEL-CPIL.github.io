"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 42 using Strassen on 2x2 blocks
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set U, V, W entries
    def add_u(row, col, val):
        U[row][col] += val
        
    def add_v(row, col, val):
        V[row][col] += val
        
    def add_w(row, col, val):
        W[row][col] += val

    # Strassen coefficients for U (A side)
    # A1 (cols 0,1), A2 (cols 2,3)
    # A1 indices: 0,1,4,5 (i*4+j)
    # A2 indices: 2,3,6,7
    
    # Strassen coefficients for V (B side)
    # B1b (rows 0,1, cols 2b, 2b+1), B2b (rows 2,3, cols 2b, 2b+1)
    # B1b indices: 2b, 2b+1, 6+2b, 7+2b
    # B2b indices: 12+2b, 13+2b, 18+2b, 19+2b

    for b in range(3):
        base_col = b * 14
        # P1 = A1 * B1b (cols 0-6)
        # c0: (A00+A11)(B00+B11)
        add_u(0, base_col+0, 1); add_u(5, base_col+0, 1)
        add_v(2*b, base_col+0, 1); add_v(6+2*b, base_col+0, 1)
        
        # c1: (A10+A11)B00
        add_u(4, base_col+1, 1); add_u(5, base_col+1, 1)
        add_v(2*b, base_col+1, 1)
        
        # c2: A00(B10-B11)
        add_u(0, base_col+2, 1)
        add_v(6+2*b, base_col+2, 1); add_v(7+2*b, base_col+2, -1)
        
        # c3: A11(B00-B01)
        add_u(5, base_col+3, 1)
        add_v(2*b, base_col+3, 1); add_v(2*b+1, base_col+3, -1)
        
        # c4: (A00+A01)B11
        add_u(0, base_col+4, 1); add_u(1, base_col+4, 1)
        add_v(7+2*b, base_col+4, 1)
        
        # c5: (A10-A00)(B00+B01)
        add_u(4, base_col+5, 1); add_u(0, base_col+5, -1)
        add_v(2*b, base_col+5, 1); add_v(2*b+1, base_col+5, 1)
        
        # c6: (A01-A11)(B10+B11)
        add_u(1, base_col+6, 1); add_u(5, base_col+6, -1)
        add_v(6+2*b, base_col+6, 1); add_v(7+2*b, base_col+6, 1)
        
        # P2 = A2 * B2b (cols 7-13)
        # c7: (A200+A211)(B200+B211)
        add_u(2, base_col+7, 1); add_u(7, base_col+7, 1)
        add_v(12+2*b, base_col+7, 1); add_v(19+2*b, base_col+7, 1)
        
        # c8: (A210+A211)B200
        add_u(6, base_col+8, 1); add_u(7, base_col+8, 1)
        add_v(12+2*b, base_col+8, 1)
        
        # c9: A200(B210-B211)
        add_u(2, base_col+9, 1)
        add_v(18+2*b, base_col+9, 1); add_v(19+2*b, base_col+9, -1)
        
        # c10: A211(B200-B201)
        add_u(7, base_col+10, 1)
        add_v(12+2*b, base_col+10, 1); add_v(13+2*b, base_col+10, -1)
        
        # c11: (A200+A201)B211
        add_u(2, base_col+11, 1); add_u(3, base_col+11, 1)
        add_v(19+2*b, base_col+11, 1)
        
        # c12: (A210-A200)(B200+B201)
        add_u(6, base_col+12, 1); add_u(2, base_col+12, -1)
        add_v(12+2*b, base_col+12, 1); add_v(13+2*b, base_col+12, 1)
        
        # c13: (A201-A211)(B210+B211)
        add_u(3, base_col+13, 1); add_u(7, base_col+13, -1)
        add_v(18+2*b, base_col+13, 1); add_v(19+2*b, base_col+13, 1)
        
        # W reconstruction for block b
        # Rows in W: k*n + i. n=2.
        # k=2b, i=0 -> 4b
        # k=2b, i=1 -> 4b+1
        # k=2b+1, i=0 -> 4b+2
        # k=2b+1, i=1 -> 4b+3
        
        # C_0, 2b = P1_00 + P2_00
        # P1_00 = c0 + c3 - c4 + c6
        # P2_00 = c7 + c10 - c11 + c13
        add_w(4*b, base_col+0, 1)
        add_w(4*b, base_col+3, 1)
        add_w(4*b, base_col+4, -1)
        add_w(4*b, base_col+6, 1)
        add_w(4*b, base_col+7, 1)
        add_w(4*b, base_col+10, 1)
        add_w(4*b, base_col+11, -1)
        add_w(4*b, base_col+13, 1)
        
        # C_0, 2b+1 = P1_01 + P2_01
        # P1_01 = c2 + c4
        # P2_01 = c9 + c11
        add_w(4*b+1, base_col+2, 1)
        add_w(4*b+1, base_col+4, 1)
        add_w(4*b+1, base_col+9, 1)
        add_w(4*b+1, base_col+11, 1)
        
        # C_1, 2b = P1_10 + P2_10
        # P1_10 = c1 + c3
        # P2_10 = c8 + c10
        add_w(4*b+2, base_col+1, 1)
        add_w(4*b+2, base_col+3, 1)
        add_w(4*b+2, base_col+8, 1)
        add_w(4*b+2, base_col+10, 1)
        
        # C_1, 2b+1 = P1_11 + P2_11
        # P1_11 = c0 - c1 + c2 + c5
        # P2_11 = c7 - c8 + c9 + c12
        add_w(4*b+3, base_col+0, 1)
        add_w(4*b+3, base_col+1, -1)
        add_w(4*b+3, base_col+2, 1)
        add_w(4*b+3, base_col+5, 1)
        add_w(4*b+3, base_col+7, 1)
        add_w(4*b+3, base_col+8, -1)
        add_w(4*b+3, base_col+9, 1)
        add_w(4*b+3, base_col+12, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
