"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # Helper for Strassen block
    # u_a, u_b, u_c, u_d are U indices for A = [[a,b],[c,d]]
    # v_e, v_f, v_g, v_h are V indices for B = [[e,f],[g,h]]
    # w_00, w_01, w_10, w_11 are W indices for C = [[C00,C01],[C10,C11]]
    def add_strassen(u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_00, w_01, w_10, w_11):
        nonlocal col
        # M1 = (a+d)(e+h) -> C00 +, C11 +
        U[u_a][col] = 1; U[u_d][col] = 1
        V[v_e][col] = 1; V[v_h][col] = 1
        W[w_00][col] = 1; W[w_11][col] = 1
        col += 1
        
        # M2 = (c+d)e -> C10 +, C11 -
        U[u_c][col] = 1; U[u_d][col] = 1
        V[v_e][col] = 1
        W[w_10][col] = 1; W[w_11][col] = -1
        col += 1
        
        # M3 = a(f-h) -> C01 +, C11 +
        U[u_a][col] = 1
        V[v_f][col] = 1; V[v_h][col] = -1
        W[w_01][col] = 1; W[w_11][col] = 1
        col += 1
        
        # M4 = d(g-e) -> C00 +, C10 +
        U[u_d][col] = 1
        V[v_g][col] = 1; V[v_e][col] = -1
        W[w_00][col] = 1; W[w_10][col] = 1
        col += 1
        
        # M5 = (a+b)h -> C00 -, C01 +
        U[u_a][col] = 1; U[u_b][col] = 1
        V[v_h][col] = 1
        W[w_00][col] = -1; W[w_01][col] = 1
        col += 1
        
        # M6 = (c-a)(e+f) -> C11 +
        U[u_c][col] = 1; U[u_a][col] = -1
        V[v_e][col] = 1; V[v_f][col] = 1
        W[w_11][col] = 1
        col += 1
        
        # M7 = (b-d)(g+h) -> C00 +, C10 -
        U[u_b][col] = 1; U[u_d][col] = -1
        V[v_g][col] = 1; V[v_h][col] = 1
        W[w_00][col] = 1; W[w_10][col] = -1
        col += 1

    # Indices mapping
    # U: i*m + j. i in [0,2), j in [0,3)
    # A = [[A00, A01, A02], [A10, A11, A12]]
    # u_a = A00 = 0*3+0 = 0
    # u_b = A01 = 0*3+1 = 1
    # u_c = A10 = 1*3+0 = 3
    # u_d = A11 = 1*3+1 = 4
    # A3 column:
    # u_30 = A02 = 0*3+2 = 2
    # u_31 = A12 = 1*3+2 = 5
    
    # V: j*p + k. j in [0,3), k in [0,5)
    # B = [[B00, B01, B02, B03, B04],
    #      [B10, B11, B12, B13, B14],
    #      [B20, B21, B22, B23, B24]]
    # B12 block (j=0,1):
    # v_e (B00) = 0*5+0 = 0
    # v_f (B01) = 0*5+1 = 1
    # v_g (B10) = 1*5+0 = 5
    # v_h (B11) = 1*5+1 = 6
    # B12 block 2 (j=0,1, k=2,3):
    # v_e2 (B02) = 0*5+2 = 2
    # v_f2 (B03) = 0*5+3 = 3
    # v_g2 (B12) = 1*5+2 = 7
    # v_h2 (B13) = 1*5+3 = 8
    # B12 block 3 (j=0,1, k=4):
    # v_x (B04) = 0*5+4 = 4
    # v_y (B14) = 1*5+4 = 9
    # B3 row (j=2):
    # v_z0 (B20) = 2*5+0 = 10
    # v_z1 (B21) = 2*5+1 = 11
    # v_z2 (B22) = 2*5+2 = 12
    # v_z3 (B23) = 2*5+3 = 13
    # v_z4 (B24) = 2*5+4 = 14

    # W: k*n + i. k in [0,5), i in [0,2)
    # C = [[C00, C01, C02, C03, C04],
    #      [C10, C11, C12, C13, C14]]
    # w_00 (C00) = 0*2+0 = 0
    # w_01 (C01) = 1*2+0 = 2 ? No. k*n + i.
    # k=0, i=0 -> 0
    # k=1, i=0 -> 2
    # k=2, i=0 -> 4
    # k=3, i=0 -> 6
    # k=4, i=0 -> 8
    # k=0, i=1 -> 1
    # k=1, i=1 -> 3
    # k=2, i=1 -> 5
    # k=3, i=1 -> 7
    # k=4, i=1 -> 9
    
    # Block 1: A12 * B12_1 (k=0,1)
    # w_00 = 0, w_01 = 2, w_10 = 1, w_11 = 3
    add_strassen(0, 1, 3, 4, 0, 1, 5, 6, 0, 2, 1, 3)
    
    # Block 2: A12 * B12_2 (k=2,3)
    # w_02 = 4, w_03 = 6, w_12 = 5, w_13 = 7
    add_strassen(0, 1, 3, 4, 2, 3, 7, 8, 4, 6, 5, 7)
    
    # Block 3: A12 * B12_3 (k=4)
    # Standard 2x2x1
    # w_04 = 8, w_14 = 9
    # a*x (u=0, v=4, w=8)
    U[0][col] = 1; V[4][col] = 1; W[8][col] = 1; col += 1
    # b*y (u=1, v=9, w=8)
    U[1][col] = 1; V[9][col] = 1; W[8][col] = 1; col += 1
    # c*x (u=3, v=4, w=9)
    U[3][col] = 1; V[4][col] = 1; W[9][col] = 1; col += 1
    # d*y (u=4, v=9, w=9)
    U[4][col] = 1; V[9][col] = 1; W[9][col] = 1; col += 1
    
    # Block 4: A3 * B3 (k=0..4)
    # u_30 = 2, u_31 = 5
    # v_z0..v_z4 = 10..14
    # w_0k = 2k, w_1k = 2k+1
    for k in range(5):
        # u_30 * z_k -> w_0k
        U[2][col] = 1; V[10+k][col] = 1; W[2*k][col] = 1; col += 1
        # u_31 * z_k -> w_1k
        U[5][col] = 1; V[10+k][col] = 1; W[2*k+1][col] = 1; col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
