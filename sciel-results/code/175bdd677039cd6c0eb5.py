"""Matrix multiplication 3x4 * 4x5 with rank-58 decomposition using Strassen."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    # A is n x m (3x4), B is m x p (4x5), C is n x p (3x5)
    # U: (n*m, rank) = (12, rank) - indexed by i*m+j for A[i][j]
    # V: (m*p, rank) = (20, rank) - indexed by j*p+k for B[j][k]
    # W: (p*n, rank) = (15, rank) - indexed by k*n+i for C[i][k] (trace convention)
    
    # Strategy:
    # Split A into A_top (row 0, 1x4) and A_bot (rows 1-2, 2x4)
    # Split B into B_left (cols 0-1, 4x2) and B_right (cols 2-4, 4x3)
    # 
    # C = A*B where:
    #   C[0][k] = A[0][0]*B[0][k] + A[0][1]*B[1][k] + A[0][2]*B[2][k] + A[0][3]*B[3][k]
    #   C[1][k] = A[1][0]*B[0][k] + A[1][1]*B[1][k] + A[1][2]*B[2][k] + A[1][3]*B[3][k]
    #   C[2][k] = A[2][0]*B[0][k] + A[2][1]*B[1][k] + A[2][2]*B[2][k] + A[2][3]*B[3][k]
    #
    # Block decomposition:
    # A_top * B: 1x4 * 4x5, rank 20 (schoolbook)
    # A_bot * B: 2x4 * 4x5
    #   Split B into B_left (4x2) and B_right (4x3)
    #   A_bot * B_left: 2x4 * 4x2, split A_bot into cols:
    #     A_bot_left (2x2) * B_left (4x2): 
    #       Further split B_left into B_ll (2x2, rows 0-1) and B_lr (2x2, rows 2-3)
    #       A_bot_left * B_ll: 2x2 * 2x2, rank 7 (Strassen)
    #       A_bot_left * B_lr: 2x2 * 2x2, rank 7 (Strassen)
    #     A_bot_right (2x2) * B_left (4x2):
    #       A_bot_right * B_ll: 2x2 * 2x2, rank 7 (Strassen)
    #       A_bot_right * B_lr: 2x2 * 2x2, rank 7 (Strassen)
    #     Total A_bot * B_left: 28
    #   A_bot * B_right: 2x4 * 4x3, schoolbook: 2*4*3 = 24
    #   Total A_bot * B: 28 + 24 = 52
    # Grand total: 20 + 52 = 72. Too high.
    #
    # Let me try a different split:
    # A (3x4) = [A1 (3x2) | A2 (3x2)]
    # B (4x5) = [[B11 (2x2), B12 (2x3)], [B21 (2x2), B22 (2x3)]]
    #
    # C = A1 * [B11, B12] + A2 * [B21, B22]
    #
    # A1 * B1: 3x2 * 2x5
    #   Split A1 into A1_top (1x2) and A1_bot (2x2)
    #   A1_top * B1: 1x2 * 2x5, rank 10 (schoolbook)
    #   A1_bot * B1: 2x2 * 2x5
    #     Split B1 into B1a (2x2) and B1b (2x3)
    #     A1_bot * B1a: 2x2 * 2x2, rank 7 (Strassen)
    #     A1_bot * B1b: 2x2 * 2x3, rank 8 (known result)
    #     Total: 15
    #   Total A1 * B1: 25
    #
    # A2 * B2: 3x2 * 2x5, same structure, rank 25
    #
    # Grand total: 50
    #
    # But I need to verify rank(2x2 * 2x3) = 8. Let me use schoolbook for safety: 12.
    # Then total: 2*(10+7+12) = 2*29 = 58.
    
    rank = 58
    U = [[0] * rank for _ in range(n * m)]  # 12 x 58
    V = [[0] * rank for _ in range(m * p)]  # 20 x 58
    W = [[0] * rank for _ in range(p * n)]  # 15 x 58
    
    col = 0
    
    # U indices: U[i*m+j] for A[i][j], i in 0..2, j in 0..3
    # A1 (cols 0-1): U[0*4+0]=U[0], U[0*4+1]=U[1], U[1*4+0]=U[4], U[1*4+1]=U[5], U[2*4+0]=U[8], U[2*4+1]=U[9]
    # A2 (cols 2-3): U[0*4+2]=U[2], U[0*4+3]=U[3], U[1*4+2]=U[6], U[1*4+3]=U[7], U[2*4+2]=U[10], U[2*4+3]=U[11]
    
    # V indices: V[j*p+k] for B[j][k], j in 0..3, k in 0..4
    # B1 (rows 0-1): V[0*5+0]=V[0], V[0*5+1]=V[1], V[0*5+2]=V[2], V[0*5+3]=V[3], V[0*5+4]=V[4], V[1*5+0]=V[5], V[1*5+1]=V[6], V[1*5+2]=V[7], V[1*5+3]=V[8], V[1*5+4]=V[9]
    # B2 (rows 2-3): V[2*5+0]=V[10], V[2*5+1]=V[11], V[2*5+2]=V[12], V[2*5+3]=V[13], V[2*5+4]=V[14], V[3*5+0]=V[15], V[3*5+1]=V[16], V[3*5+2]=V[17], V[3*5+3]=V[18], V[3*5+4]=V[19]
    
    # W indices: W[k*n+i] for C[i][k], i in 0..2, k in 0..4
    # C[0][k]: W[k*3+0], C[1][k]: W[k*3+1], C[2][k]: W[k*3+2]
    
    # Helper function to add a schoolbook term
    def add_term(u_idx, v_idx, w_idx, coefficient=1):
        nonlocal col
        if coefficient != 0:
            U[u_idx][col] += coefficient
            V[v_idx][col] += coefficient
            W[w_idx][col] += coefficient
        col += 1
    
    # Helper function for Strassen 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Result C = [[c00, c01], [c10, c11]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    def strassen_2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00_w, c01_w, c10_w, c11_w):
        """Add Strassen decomposition for 2x2*2x2.
        a00, a01, a10, a11: U indices for A entries
        b00, b01, b10, b11: V indices for B entries
        c00_w, c01_w, c10_w, c11_w: W indices for C entries
        """
        nonlocal col
        
        # P1 = (a+d)(e+h)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00_w][col] += 1
        W[c11_w][col] += 1
        col += 1
        
        # P2 = (c+d)e
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10_w][col] += 1
        col += 1
        
        # P3 = a(f-h)
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] += -1
        W[c01_w][col] += 1
        W[c11_w][col] += 1
        col += 1
        
        # P4 = d(g-e)
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] += -1
        W[c00_w][col] += 1
        W[c10_w][col] += 1
        col += 1
        
        # P5 = (a+b)h
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00_w][col] += -1
        W[c01_w][col] += 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[a10][col] += 1
        U[a00][col] += -1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11_w][col] += 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[a01][col] += 1
        U[a11][col] += -1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00_w][col] += 1
        col += 1
    
    # Part 1: A1_top (row 0, cols 0-1) * B1 (rows 0-1, all 5 cols)
    # 1x2 * 2x5, schoolbook: 10 terms
    for k in range(5):
        add_term(0, k, k * 3 + 0, 1)  # A[0][0]*B[0][k] -> C[0][k]
        add_term(1, 5 + k, k * 3 + 0, 1)  # A[0][1]*B[1][k] -> C[0][k]
    
    # Part 2: A1_bot (rows 1-2, cols 0-1) * B1a (rows 0-1, cols 0-1)
    # 2x2 * 2x2, Strassen: 7 terms
    # A: a00=U[4]=A[1][0], a01=U[5]=A[1][1], a10=U[8]=A[2][0], a11=U[9]=A[2][1]
    # B: b00=V[0]=B[0][0], b01=V[1]=B[0][1], b10=V[5]=B[1][0], b11=V[6]=B[1][1]
    # C: c00=C[1][0]=W[0*3+1]=W[1], c01=C[1][1]=W[1*3+1]=W[4], c10=C[2][0]=W[0*3+2]=W[2], c11=C[2][1]=W[1*3+2]=W[5]
    strassen_2x2(4, 5, 8, 9, 0, 1, 5, 6, 1, 4, 2, 5)
    
    # Part 3: A1_bot (rows 1-2, cols 0-1) * B1b (rows 0-1, cols 2-4)
    # 2x2 * 2x3, schoolbook: 12 terms
    for k in range(2, 5):
        add_term(4, k, k * 3 + 1, 1)  # A[1][0]*B[0][k] -> C[1][k]
        add_term(5, 5 + k, k * 3 + 1, 1)  # A[1][1]*B[1][k] -> C[1][k]
        add_term(8, k, k * 3 + 2, 1)  # A[2][0]*B[0][k] -> C[2][k]
        add_term(9, 5 + k, k * 3 + 2, 1)  # A[2][1]*B[1][k] -> C[2][k]
    
    # Part 4: A2_top (row 0, cols 2-3) * B2 (rows 2-3, all 5 cols)
    # 1x2 * 2x5, schoolbook: 10 terms
    for k in range(5):
        add_term(2, 10 + k, k * 3 + 0, 1)  # A[0][2]*B[2][k] -> C[0][k]
        add_term(3, 15 + k, k * 3 + 0, 1)  # A[0][3]*B[3][k] -> C[0][k]
    
    # Part 5: A2_bot (rows 1-2, cols 2-3) * B2a (rows 2-3, cols 0-1)
    # 2x2 * 2x2, Strassen: 7 terms
    # A: a00=U[6]=A[1][2], a01=U[7]=A[1][3], a10=U[10]=A[2][2], a11=U[11]=A[2][3]
    # B: b00=V[10]=B[2][0], b01=V[11]=B[2][1], b10=V[15]=B[3][0], b11=V[16]=B[3][1]
    # C: c00=C[1][0]=W[1], c01=C[1][1]=W[4], c10=C[2][0]=W[2], c11=C[2][1]=W[5]
    strassen_2x2(6, 7, 10, 11, 10, 11, 15, 16, 1, 4, 2, 5)
    
    # Part 6: A2_bot (rows 1-2, cols 2-3) * B2b (rows 2-3, cols 2-4)
    # 2x2 * 2x3, schoolbook: 12 terms
    for k in range(2, 5):
        add_term(6, 12 + (k-2), k * 3 + 1, 1)  # A[1][2]*B[2][k] -> C[1][k]
        add_term(7, 17 + (k-2), k * 3 + 1, 1)  # A[1][3]*B[3][k] -> C[1][k]
        add_term(10, 12 + (k-2), k * 3 + 2, 1)  # A[2][2]*B[2][k] -> C[2][k]
        add_term(11, 17 + (k-2), k * 3 + 2, 1)  # A[2][3]*B[3][k] -> C[2][k]
    
    # Total: 10 + 7 + 12 + 10 + 7 + 12 = 58
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
