"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 28 decomposition using Strassen's algorithm on 2x2 blocks.
    # A (2x4) is split into A_L (2x2) and A_R (2x2).
    # B (4x4) is split into 4 blocks of 2x2: B_LL, B_LR, B_RL, B_RR.
    # C (2x4) is split into C_L (2x2) and C_R (2x2).
    # C_L = A_L B_LL + A_R B_RL
    # C_R = A_L B_LR + A_R B_RR
    
    # We use the linear combination trick:
    # C_L = 0.5 * X * (B_LL + B_RL) + 0.5 * Y * (B_LL - B_RL)
    # C_R = 0.5 * X * (B_LR + B_RR) + 0.5 * Y * (B_LR - B_RR)
    # where X = A_L + A_R, Y = A_L - A_R.
    
    # Each term X * R and Y * R is a 2x2 matrix multiplication.
    # We use Strassen (rank 7) for each.
    # Total 4 Strassen multiplications -> 28 multiplications.
    
    n, m, p = DIMENSIONS
    rank = 28
    
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen decomposition for A (2x2) * B (2x2) -> C (2x2)
    # A indices in U: row_offset + 0..3
    # B indices in V: col_offset + 0..3
    # C indices in W: out_offset + 0..3
    # Strassen factors for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We define the 7 multiplications for one 2x2 block.
    # The indices for a 2x2 matrix [a b; c d] are 0, 1, 2, 3.
    
    def add_strassen(u_row_base, v_row_base, w_row_base, start_col):
        # u_row_base: row index in U for the first element of A (2x2)
        # v_row_base: row index in V for the first element of B (2x2)
        # w_row_base: row index in W for the first element of C (2x2)
        
        # A is 2x2. Rows in U: u_row_base, u_row_base+1, u_row_base+2, u_row_base+3
        # A00 at u_row_base, A01 at u_row_base+1, A10 at u_row_base+2, A11 at u_row_base+3
        # Note: U is indexed by [i*m + j]. For 2x2 block, i=0,1, j=0,1.
        # If A_L is top-left 2x2 of A (2x4), its indices are 0,1,2,3.
        # If A_R is top-right 2x2 of A (2x4), its indices are 4,5,6,7.
        
        # B is 2x2. Rows in V: v_row_base .. v_row_base+3
        # B00, B01, B10, B11.
        
        # C is 2x2. Rows in W: w_row_base .. w_row_base+3
        # C00, C01, C10, C11.
        # W is indexed by [k*n + i]. For 2x2 block of C (2x4), 
        # C_L is cols 0,1. C_R is cols 2,3.
        # C_L indices: 0,1,2,3. C_R indices: 4,5,6,7.
        
        # M1
        U[u_row_base][start_col] += 1   # A00
        U[u_row_base+3][start_col] += 1 # A11
        V[v_row_base][start_col] += 1   # B00
        V[v_row_base+1][start_col] += 1 # B01
        V[v_row_base+2][start_col] += 1 # B10
        V[v_row_base+3][start_col] += 1 # B11
        W[w_row_base][start_col] += 1   # C00
        W[w_row_base+1][start_col] += 1 # C01
        W[w_row_base+2][start_col] += 1 # C10
        W[w_row_base+3][start_col] += 1 # C11
        # Wait, Strassen output is linear combination of Ms.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So we need to add coefficients to W.
        # Let's define the 7 Ms and their contributions to C.
        
        # M1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1. V: B00=1, B11=1.
        # W: C00=1, C11=1.
        
        # M2: (A10+A11)B00
        # U: A10=1, A11=1. V: B00=1.
        # W: C10=1, C11=-1.
        
        # M3: A00(B01-B11)
        # U: A00=1. V: B01=1, B11=-1.
        # W: C01=1, C11=1.
        
        # M4: A11(B10-B00)
        # U: A11=1. V: B10=1, B00=-1.
        # W: C00=1, C10=1.
        
        # M5: (A00+A01)B11
        # U: A00=1, A01=1. V: B11=1.
        # W: C00=-1, C01=1.
        
        # M6: (A10-A00)(B00+B01)
        # U: A10=1, A00=-1. V: B00=1, B01=1.
        # W: C11=1.
        
        # M7: (A01-A11)(B10+B11)
        # U: A01=1, A11=-1. V: B10=1, B11=1.
        # W: C00=1, C10=-1.

        # We will add these 7 multiplications.
        
        # M1
        U[u_row_base][start_col] += 1
        U[u_row_base+3][start_col] += 1
        V[v_row_base][start_col] += 1
        V[v_row_base+3][start_col] += 1
        W[w_row_base][start_col] += 1
        W[w_row_base+3][start_col] += 1
        
        # M2
        U[u_row_base+2][start_col+1] += 1
        U[u_row_base+3][start_col+1] += 1
        V[v_row_base][start_col+1] += 1
        W[w_row_base+2][start_col+1] += 1
        W[w_row_base+3][start_col+1] -= 1
        
        # M3
        U[u_row_base][start_col+2] += 1
        V[v_row_base+1][start_col+2] += 1
        V[v_row_base+3][start_col+2] -= 1
        W[w_row_base+1][start_col+2] += 1
        W[w_row_base+3][start_col+2] += 1
        
        # M4
        U[u_row_base+3][start_col+3] += 1
        V[v_row_base+2][start_col+3] += 1
        V[v_row_base][start_col+3] -= 1
        W[w_row_base][start_col+3] += 1
        W[w_row_base+2][start_col+3] += 1
        
        # M5
        U[u_row_base][start_col+4] += 1
        U[u_row_base+1][start_col+4] += 1
        V[v_row_base+3][start_col+4] += 1
        W[w_row_base][start_col+4] -= 1
        W[w_row_base+1][start_col+4] += 1
        
        # M6
        U[u_row_base+2][start_col+5] += 1
        U[u_row_base][start_col+5] -= 1
        V[v_row_base][start_col+5] += 1
        V[v_row_base+1][start_col+5] += 1
        W[w_row_base+3][start_col+5] += 1
        
        # M7
        U[u_row_base+1][start_col+6] += 1
        U[u_row_base+3][start_col+6] -= 1
        V[v_row_base+2][start_col+6] += 1
        V[v_row_base+3][start_col+6] += 1
        W[w_row_base][start_col+6] += 1
        W[w_row_base+2][start_col+6] -= 1

    # We need to form X = A_L + A_R and Y = A_L - A_R.
    # This means we cannot just use U rows for A_L and A_R directly in Strassen.
    # We need to create new "virtual" A matrices X and Y.
    # U for X:
    # X00 = A_L00 + A_R00
    # X01 = A_L01 + A_R01
    # X10 = A_L10 + A_R10
    # X11 = A_L11 + A_R11
    
    # U for Y:
    # Y00 = A_L00 - A_R00
    # ...
    
    # This complicates the U matrix construction.
    # Instead of adding Strassen directly, we define the U factors for X and Y.
    
    # Let's restart the construction with explicit U, V, W for the 4 Strassen blocks.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: X * (B_LL + B_RL) -> contributes to C_L
    # X is 2x2. B_LL + B_RL is 2x2.
    # U indices for X:
    # X00 = A00 + A04 (A_L00 + A_R00)
    # X01 = A01 + A05
    # X10 = A02 + A06
    # X11 = A03 + A07
    
    # V indices for R1 = B_LL + B_RL:
    # B_LL is rows 0-3 of V. B_RL is rows 8-11 of V (since B is 4x4, row 2, col 0-1).
    # B_LL: 0,1,2,3
    # B_RL: 8,9,10,11
    
    # W indices for C_L: 0-3.
    
    # We add 7 multiplications for Block 1.
    # start_col = 0
    
    # M1_1: (X00+X11)(R1_00+R1_11)
    # X00+X11 = A00+A04+A03+A07
    # R1_00+R1_11 = B00+B08 + B03+B11
    
    # This is getting tedious to hardcode manually.
    # I will use a helper to add a generic 2x2 multiplication with given linear combinations.
    
    def add_mult(u_coefs, v_coefs, w_coefs, col):
        # u_coefs: list of (row, coef)
        # v_coefs: list of (row, coef)
        # w_coefs: list of (row, coef)
        for r, c in u_coefs:
            U[r][col] += c
        for r, c in v_coefs:
            V[r][col] += c
        for r, c in w_coefs:
            W[r][col] += c

    # Define Strassen factors for 2x2x2
    # A rows: 0,1,2,3. B rows: 0,1,2,3. C rows: 0,1,2,3.
    
    # M1: (A0+A3)(B0+B3) -> C0=1, C3=1
    s1_u = [(0,1),(3,1)]
    s1_v = [(0,1),(3,1)]
    s1_w = [(0,1),(3,1)]
    
    # M2: (A2+A3)B0 -> C2=1, C3=-1
    s2_u = [(2,1),(3,1)]
    s2_v = [(0,1)]
    s2_w = [(2,1),(3,-1)]
    
    # M3: A0(B1-B3) -> C1=1, C3=1
    s3_u = [(0,1)]
    s3_v = [(1,1),(3,-1)]
    s3_w = [(1,1),(3,1)]
    
    # M4: A3(B2-B0) -> C0=1, C2=1
    s4_u = [(3,1)]
    s4_v = [(2,1),(0,-1)]
    s4_w = [(0,1),(2,1)]
    
    # M5: (A0+A1)B3 -> C0=-1, C1=1
    s5_u = [(0,1),(1,1)]
    s5_v = [(3,1)]
    s5_w = [(0,-1),(1,1)]
    
    # M6: (A2-A0)(B0+B1) -> C3=1
    s6_u = [(2,1),(0,-1)]
    s6_v = [(0,1),(1,1)]
    s6_w = [(3,1)]
    
    # M7: (A1-A3)(B2+B3) -> C0=1, C2=-1
    s7_u = [(1,1),(3,-1)]
    s7_v = [(2,1),(3,1)]
    s7_w = [(0,1),(2,-1)]
    
    # Block 1: X * R1
    # X rows in U:
    # X0: 0, 4
    # X1: 1, 5
    # X2: 2, 6
    # X3: 3, 7
    
    # R1 rows in V:
    # R1_0: 0, 8
    # R1_1: 1, 9
    # R1_2: 2, 10
    # R1_3: 3, 11
    
    # C_L rows in W: 0, 1, 2, 3
    
    # Map Strassen indices to actual rows
    # A0->X0, A1->X1, A2->X2, A3->X3
    # B0->R1_0, B1->R1_1, B2->R1_2, B3->R1_3
    # C0->C0, C1->C1, C2->C2, C3->C3
    
    def map_u(u_list):
        return [(r+4 if r >= 4 else r, c) for r, c in u_list] # Wrong mapping
    # Correct mapping:
    # X0 is sum of U[0] and U[4].
    # So for any coef on A0, we add to U[0] and U[4].
    
    def expand_u(u_list):
        res = []
        for r, c in u_list:
            # r is 0,1,2,3 corresponding to X0,X1,X2,X3
            # X0: U[0], U[4]
            # X1: U[1], U[5]
            # X2: U[2], U[6]
            # X3: U[3], U[7]
            res.append((r, c))
            res.append((r+4, c))
        return res
        
    def expand_v(v_list, base1, base2):
        # v_list indices 0,1,2,3
        # base1: 0, base2: 8
        res = []
        for r, c in v_list:
            res.append((base1+r, c))
            res.append((base2+r, c))
        return res
        
    def expand_w(w_list, base):
        res = []
        for r, c in w_list:
            res.append((base+r, c))
        return res
        
    # Block 1: X * (B_LL + B_RL)
    # X is sum of A_L (0-3) and A_R (4-7).
    # R1 is sum of B_LL (0-3) and B_RL (8-11).
    
    col = 0
    # M1
    add_mult(expand_u(s1_u), expand_v(s1_v, 0, 8), expand_w(s1_w, 0), col)
    col += 1
    # M2
    add_mult(expand_u(s2_u), expand_v(s2_v, 0, 8), expand_w(s2_w, 0), col)
    col += 1
    # M3
    add_mult(expand_u(s3_u), expand_v(s3_v, 0, 8), expand_w(s3_w, 0), col)
    col += 1
    # M4
    add_mult(expand_u(s4_u), expand_v(s4_v, 0, 8), expand_w(s4_w, 0), col)
    col += 1
    # M5
    add_mult(expand_u(s5_u), expand_v(s5_v, 0, 8), expand_w(s5_w, 0), col)
    col += 1
    # M6
    add_mult(expand_u(s6_u), expand_v(s6_v, 0, 8), expand_w(s6_w, 0), col)
    col += 1
    # M7
    add_mult(expand_u(s7_u), expand_v(s7_v, 0, 8), expand_w(s7_w, 0), col)
    col += 1
    
    # Block 2: Y * (B_LL - B_RL)
    # Y is A_L - A_R.
    # R2 is B_LL - B_RL.
    
    # expand_u for Y:
    def expand_u_y(u_list):
        res = []
        for r, c in u_list:
            res.append((r, c))
            res.append((r+4, -c))
        return res
        
    def expand_v_y(v_list, base1, base2):
        res = []
        for r, c in v_list:
            res.append((base1+r, c))
            res.append((base2+r, -c))
        return res
        
    # M1
    add_mult(expand_u_y(s1_u), expand_v_y(s1_v, 0, 8), expand_w(s1_w, 0), col)
    col += 1
    # M2
    add_mult(expand_u_y(s2_u), expand_v_y(s2_v, 0, 8), expand_w(s2_w, 0), col)
    col += 1
    # M3
    add_mult(expand_u_y(s3_u), expand_v_y(s3_v, 0, 8), expand_w(s3_w, 0), col)
    col += 1
    # M4
    add_mult(expand_u_y(s4_u), expand_v_y(s4_v, 0, 8), expand_w(s4_w, 0), col)
    col += 1
    # M5
    add_mult(expand_u_y(s5_u), expand_v_y(s5_v, 0, 8), expand_w(s5_w, 0), col)
    col += 1
    # M6
    add_mult(expand_u_y(s6_u), expand_v_y(s6_v, 0, 8), expand_w(s6_w, 0), col)
    col += 1
    # M7
    add_mult(expand_u_y(s7_u), expand_v_y(s7_v, 0, 8), expand_w(s7_w, 0), col)
    col += 1
    
    # Block 3: X * (B_LR + B_RR)
    # X is A_L + A_R.
    # R3 is B_LR (4-7) + B_RR (12-15).
    
    # M1
    add_mult(expand_u(s1_u), expand_v(s1_v, 4, 12), expand_w(s1_w, 4), col)
    col += 1
    # M2
    add_mult(expand_u(s2_u), expand_v(s2_v, 4, 12), expand_w(s2_w, 4), col)
    col += 1
    # M3
    add_mult(expand_u(s3_u), expand_v(s3_v, 4, 12), expand_w(s3_w, 4), col)
    col += 1
    # M4
    add_mult(expand_u(s4_u), expand_v(s4_v, 4, 12), expand_w(s4_w, 4), col)
    col += 1
    # M5
    add_mult(expand_u(s5_u), expand_v(s5_v, 4, 12), expand_w(s5_w, 4), col)
    col += 1
    # M6
    add_mult(expand_u(s6_u), expand_v(s6_v, 4, 12), expand_w(s6_w, 4), col)
    col += 1
    # M7
    add_mult(expand_u(s7_u), expand_v(s7_v, 4, 12), expand_w(s7_w, 4), col)
    col += 1
    
    # Block 4: Y * (B_LR - B_RR)
    # Y is A_L - A_R.
    # R4 is B_LR (4-7) - B_RR (12-15).
    
    # M1
    add_mult(expand_u_y(s1_u), expand_v_y(s1_v, 4, 12), expand_w(s1_w, 4), col)
    col += 1
    # M2
    add_mult(expand_u_y(s2_u), expand_v_y(s2_v, 4, 12), expand_w(s2_w, 4), col)
    col += 1
    # M3
    add_mult(expand_u_y(s3_u), expand_v_y(s3_v, 4, 12), expand_w(s3_w, 4), col)
    col += 1
    # M4
    add_mult(expand_u_y(s4_u), expand_v_y(s4_v, 4, 12), expand_w(s4_w, 4), col)
    col += 1
    # M5
    add_mult(expand_u_y(s5_u), expand_v_y(s5_v, 4, 12), expand_w(s5_w, 4), col)
    col += 1
    # M6
    add_mult(expand_u_y(s6_u), expand_v_y(s6_v, 4, 12), expand_w(s6_w, 4), col)
    col += 1
    # M7
    add_mult(expand_u_y(s7_u), expand_v_y(s7_v, 4, 12), expand_w(s7_w, 4), col)
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
