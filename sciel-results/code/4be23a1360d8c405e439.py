"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Outer Strassen coefficients for 4x4 blocks (2x2 blocks of 2x2 matrices)
    # A_blocks[m][u][v] is coefficient for block A[u][v] in outer term m
    # u, v in {0, 1}
    A_outer = [
        [[1, 0], [0, 1]], # M1: A00 + A11
        [[0, 0], [1, 1]], # M2: A10 + A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00 + A01
        [[0, -1], [1, 0]],# M6: A10 - A00
        [[0, 1], [0, -1]] # M7: A01 - A11
    ]
    
    B_outer = [
        [[1, 0], [0, 1]], # M1: B00 + B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 1], [0, -1]],# M3: B01 - B11
        [[0, -1], [1, 0]],# M4: B10 - B00
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00 + B01
        [[0, 0], [1, 1]]  # M7: B10 + B11
    ]

    # Inner Strassen products for 2x2 matrices
    # Each product is defined by linear forms for A and B parts
    # a_forms[n][r][c] coefficient for A[r][c]
    # b_forms[n][r][c] coefficient for B[r][c]
    # r, c in {0, 1}
    
    def make_form(r0c0, r0c1, r1c0, r1c1):
        return [[r0c0, r0c1], [r1c0, r1c1]]

    A_inner = [
        make_form(1, 0, 0, 1),   # P0: a00 + a11
        make_form(0, 0, 1, 1),   # P1: a10 + a11
        make_form(1, 0, 0, 0),   # P2: a00
        make_form(0, 0, 0, 1),   # P3: a11
        make_form(1, 1, 0, 0),   # P4: a00 + a01
        make_form(0, -1, 1, 0),  # P5: a10 - a00
        make_form(0, 1, 0, -1)   # P6: a01 - a11
    ]
    
    B_inner = [
        make_form(1, 0, 0, 1),   # P0: b00 + b11
        make_form(1, 0, 0, 0),   # P1: b00
        make_form(0, 1, 0, -1),  # P2: b01 - b11
        make_form(0, -1, 1, 0),  # P3: b10 - b00
        make_form(0, 0, 0, 1),   # P4: b11
        make_form(1, 1, 0, 0),   # P5: b00 + b01
        make_form(0, 0, 1, 1)    # P6: b10 + b11
    ]

    # Contribution of each inner product Pn to the 2x2 result block Mm
    # C_m[r][c] = sum_n W_m[r][c][n] * Pn
    # r, c in {0, 1}
    W_inner = [
        [[1, 1], [0, 0]], # P0 contributes to C00 (1), C11 (1) -> Wait, standard Strassen:
        # C00 = P0 + P3 - P4 + P6
        # C01 = P2 + P4
        # C10 = P1 + P3
        # C11 = P0 - P1 + P2 + P5
        # Let's map this to W_inner[n][r][c]
    ]
    
    # Correct W_inner mapping:
    # n=0 (P0): C00+=1, C11+=1
    # n=1 (P1): C10+=1, C11-=1
    # n=2 (P2): C01+=1, C11+=1
    # n=3 (P3): C00+=1, C10+=1
    # n=4 (P4): C00-=1, C01+=1
    # n=5 (P5): C11+=1
    # n=6 (P6): C00+=1
    
    W_inner = [
        [[1, 0], [0, 1]], # P0
        [[0, 0], [1, -1]],# P1
        [[0, 1], [0, 1]], # P2
        [[1, 0], [1, 0]], # P3
        [[-1, 1], [0, 0]],# P4
        [[0, 0], [0, 1]], # P5
        [[1, 0], [0, 0]]  # P6
    ]

    # Outer Strassen contribution of Mm to final C
    # C[r][c] = sum_m W_outer[r][c][m] * Mm
    # r, c in {0, 1}
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    W_outer = [
        [[1, 0], [0, 1]], # M0
        [[0, 0], [1, -1]],# M1
        [[0, 1], [0, 1]], # M2
        [[1, 0], [1, 0]], # M3
        [[-1, 1], [0, 0]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 0], [0, 0]]  # M6
    ]

    # Now, expand to global indices
    # Global A index: i, j. Block u = i//2, v = j//2. Local ar = i%2, ac = j%2.
    # Global B index: j, k. Block u = j//2, v = k//2. Local br = j%2, bc = k%2.
    # Global C index: i, k. Block u = i//2, v = k//2. Local cr = i%2, cc = k%2.
    
    for m in range(7):
        for n in range(7):
            col = m * 7 + n
            
            # 1. Calculate U coefficients for this column
            # U[i*m + j, col]
            # The term is sum over block (u,v) of A_outer[m][u][v] * A_block[u][v]
            # A_block[u][v] element at (ar, ac) is A[i][j] where i=2u+ar, j=2v+ac
            # The inner form A_inner[n][ar][ac] multiplies this.
            
            for u in range(2):
                for v in range(2):
                    coeff_outer_A = A_outer[m][u][v]
                    if coeff_outer_A == 0: continue
                    
                    for ar in range(2):
                        for ac in range(2):
                            coeff_inner_A = A_inner[n][ar][ac]
                            if coeff_inner_A == 0: continue
                            
                            i = 2 * u + ar
                            j = 2 * v + ac
                            val = coeff_outer_A * coeff_inner_A
                            U[i * 4 + j][col] += val

            # 2. Calculate V coefficients for this column
            # V[j*p + k, col]
            # The term is sum over block (u,v) of B_outer[m][u][v] * B_block[u][v]
            # B_block[u][v] element at (br, bc) is B[j][k] where j=2u+br, k=2v+bc
            # The inner form B_inner[n][br][bc] multiplies this.
            
            for u in range(2):
                for v in range(2):
                    coeff_outer_B = B_outer[m][u][v]
                    if coeff_outer_B == 0: continue
                    
                    for br in range(2):
                        for bc in range(2):
                            coeff_inner_B = B_inner[n][br][bc]
                            if coeff_inner_B == 0: continue
                            
                            j = 2 * u + br
                            k = 2 * v + bc
                            val = coeff_outer_B * coeff_inner_B
                            V[j * 4 + k][col] += val

            # 3. Calculate W coefficients for this column
            # W[k*n + i, col]
            # The product P_n contributes to M_m at position (cr, cc) with coeff W_inner[n][cr][cc]
            # M_m contributes to C at block (u,v) position (cr, cc) with coeff W_outer[u][v][m]
            # Wait, W_outer is indexed by result block (u,v) and outer term m.
            # C_block[u][v] element (cr, cc) is C[i][k] where i=2u+cr, k=2v+cc.
            
            for cr in range(2):
                for cc in range(2):
                    coeff_inner_W = W_inner[n][cr][cc]
                    if coeff_inner_W == 0: continue
                    
                    for u in range(2):
                        for v in range(2):
                            coeff_outer_W = W_outer[u][v][m]
                            if coeff_outer_W == 0: continue
                            
                            i = 2 * u + cr
                            k = 2 * v + cc
                            val = coeff_inner_W * coeff_outer_W
                            W[k * 4 + i][col] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
