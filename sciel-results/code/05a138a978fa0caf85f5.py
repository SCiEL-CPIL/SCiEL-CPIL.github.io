"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    col = 0
    # Strassen terms for C[0:2, 0:2]
    # A[0:2, 0:2] * B[0:2, 0:2]
    # Indices in U: i*3+j, V: j*3+k, W: k*3+i
    
    # M1: a00 * b00
    add_term(0, 0, 0, 1, 1, 1, col); col += 1 # C00 += 1
    # M2: a00 * b01
    add_term(0, 1, 1, 1, 1, 1, col); col += 1 # C01 += 1
    # M3: a10 * b10
    add_term(3, 4, 3, 1, 1, 1, col); col += 1 # C10 += 1
    # M4: a11 * b11
    add_term(4, 5, 4, 1, 1, 1, col); col += 1 # C11 += 1
    # M5: (a00+a11) * (b00+b11)
    add_term(0, 0, 0, 1, 1, 1, col)
    add_term(4, 0, 0, 1, 1, 1, col)
    add_term(0, 5, 0, 1, 1, 1, col)
    add_term(4, 5, 0, 1, 1, 1, col)
    # W contributions for M5
    add_term(0, 0, 0, 0, 0, 1, col) # C00
    add_term(0, 1, 1, 0, 0, 1, col) # C01
    add_term(3, 4, 3, 0, 0, 1, col) # C10
    add_term(4, 5, 4, 0, 0, 1, col) # C11
    # Wait, I need to subtract M1, M2, M3, M4 appropriately in W.
    # C00 = M1 + M5 - M4 + M7
    # C01 = M2 + M5 - M1 + M6
    # C10 = M3 + M5 - M2 + M7
    # C11 = M4 + M5 - M3 + M6
    
    # Let's re-construct W for Strassen properly.
    # Reset W for col 0-6? No, I'll just build the full W vector for each col.
    
    # I will rewrite the loop to be cleaner.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Col 0: M1 = a00 b00
    U[0][0] = 1; V[0][0] = 1
    W[0][0] = 1   # C00
    W[1][0] = 0   # C01
    W[3][0] = 0   # C10
    W[4][0] = 0   # C11
    # Actually W index is k*n+i.
    # C00 is i=0, k=0 -> idx 0
    # C01 is i=0, k=1 -> idx 1
    # C10 is i=1, k=0 -> idx 3
    # C11 is i=1, k=1 -> idx 4
    
    # Col 1: M2 = a00 b01
    U[0][1] = 1; V[1][1] = 1
    W[0][1] = 0
    W[1][1] = 1   # C01
    W[3][1] = 0
    W[4][1] = 0
    
    # Col 2: M3 = a10 b10
    U[3][2] = 1; V[4][2] = 1
    W[0][2] = 0
    W[1][2] = 0
    W[3][2] = 1   # C10
    W[4][2] = 0
    
    # Col 3: M4 = a11 b11
    U[4][3] = 1; V[5][3] = 1
    W[0][3] = 0
    W[1][3] = 0
    W[3][3] = 0
    W[4][3] = 1   # C11
    
    # Col 4: M5 = (a00+a11)(b00+b11)
    U[0][4] = 1; U[4][4] = 1
    V[0][4] = 1; V[5][4] = 1
    W[0][4] = 1   # C00
    W[1][4] = 1   # C01
    W[3][4] = 1   # C10
    W[4][4] = 1   # C11
    
    # Col 5: M6 = (a10+a11)(b10+b11)
    U[3][5] = 1; U[4][5] = 1
    V[4][5] = 1; V[5][5] = 1
    W[0][5] = 0
    W[1][5] = 1   # C01
    W[3][5] = 0
    W[4][5] = 1   # C11
    
    # Col 6: M7 = (a00+a10)(b00+b10)
    U[0][6] = 1; U[3][6] = 1
    V[0][6] = 1; V[4][6] = 1
    W[0][6] = 1   # C00
    W[1][6] = 0
    W[3][6] = 1   # C10
    W[4][6] = 0
    
    # Now adjust W to match Strassen formulas:
    # C00 = M1 + M5 - M4 + M7
    # C01 = M2 + M5 - M1 + M6
    # C10 = M3 + M5 - M2 + M7
    # C11 = M4 + M5 - M3 + M6
    
    # Current W sums:
    # C00: M1(1) + M5(1) + M7(1) = 3. Need 1 - 0 - 0 + 1? No.
    # Target C00 coeff: M1:1, M2:0, M3:0, M4:-1, M5:1, M6:0, M7:1
    # Target C01 coeff: M1:-1, M2:1, M3:0, M4:0, M5:1, M6:1, M7:0
    # Target C10 coeff: M1:0, M2:-1, M3:1, M4:0, M5:1, M6:0, M7:1
    # Target C11 coeff: M1:0, M2:0, M3:-1, M4:1, M5:1, M6:1, M7:0
    
    # I will overwrite W[0:5, 0:7] with the correct coefficients.
    W[0][0] = 1; W[0][1] = -1; W[0][2] = 0; W[0][3] = -1; W[0][4] = 1; W[0][5] = 0; W[0][6] = 1
    W[1][0] = 0; W[1][1] = 1;  W[1][2] = 0; W[1][3] = 0;  W[1][4] = 1; W[1][5] = 1; W[1][6] = 0
    W[3][0] = 0; W[3][1] = -1; W[3][2] = 1; W[3][3] = 0;  W[3][4] = 1; W[3][5] = 0; W[3][6] = 1
    W[4][0] = 0; W[4][1] = 0;  W[4][2] = -1;W[4][3] = 1;  W[4][4] = 1; W[4][5] = 1; W[4][6] = 0
    
    # Remaining 15 terms for C02, C12, C20, C21, C22
    # C02 = a00b02 + a01b12 + a02b22
    # C12 = a10b02 + a11b12 + a12b22
    # C20 = a20b00 + a21b10 + a22b20
    # C21 = a20b01 + a21b11 + a22b21
    # C22 = a20b02 + a21b12 + a22b22
    
    terms = [
        (0, 2, 2, 1, 1, 1),   # a00 b02 -> C02
        (1, 5, 2, 1, 1, 1),   # a01 b12 -> C02
        (2, 8, 2, 1, 1, 1),   # a02 b22 -> C02
        (3, 2, 5, 1, 1, 1),   # a10 b02 -> C12
        (4, 5, 5, 1, 1, 1),   # a11 b12 -> C12
        (5, 8, 5, 1, 1, 1),   # a12 b22 -> C12
        (6, 0, 3, 1, 1, 1),   # a20 b00 -> C20
        (7, 4, 3, 1, 1, 1),   # a21 b10 -> C20
        (8, 8, 3, 1, 1, 1),   # a22 b20 -> C20
        (6, 1, 4, 1, 1, 1),   # a20 b01 -> C21
        (7, 5, 4, 1, 1, 1),   # a21 b11 -> C21
        (8, 9, 4, 1, 1, 1),   # a22 b21 -> C21
        (6, 2, 8, 1, 1, 1),   # a20 b02 -> C22
        (7, 5, 8, 1, 1, 1),   # a21 b12 -> C22
        (8, 8, 8, 1, 1, 1),   # a22 b22 -> C22
    ]
    
    for i, (u_i, v_i, w_i, u_v, v_v, w_v) in enumerate(terms):
        c = 7 + i
        U[u_i][c] = u_v
        V[v_i][c] = v_v
        W[w_i][c] = w_v
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
