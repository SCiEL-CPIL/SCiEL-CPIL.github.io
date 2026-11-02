"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank-26 decomposition using Strassen for 2x2 block and schoolbook for the rest
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U: A_00, A_01, A_02, A_10, A_11, A_12, A_20, A_21, A_22
    # V: B_00, B_01, B_02, B_10, B_11, B_12, B_20, B_21, B_22
    # W: C_00, C_10, C_20, C_01, C_11, C_21, C_02, C_12, C_22 (k*n+i)
    
    # Strassen terms for 2x2 block (A00, A01, A10, A11) and (B00, B01, B10, B11)
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1; V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = -1; W[4][0] = 1; W[1][0] = -1 # C00=C11, C01, C10 adjustments?
    # Standard Strassen:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M2 + M4
    # C10 = M1 - M3 + M5 + M7
    # C11 = M1 - M2 + M3 - M5
    
    # M2 = A11(B00+B01)
    U[4][1] = 1; V[0][1] = 1; V[1][1] = 1
    W[3][1] = 1; W[4][1] = -1
    
    # M3 = (A00+A01)B11
    U[0][2] = 1; U[1][2] = 1; V[4][2] = 1
    W[1][2] = 1; W[4][2] = -1
    
    # M4 = (A10+A11)(B01+B11)
    U[3][3] = 1; U[4][3] = 1; V[1][3] = 1; V[4][3] = 1
    W[0][3] = -1; W[3][3] = 1; W[1][3] = -1; W[4][3] = 1
    
    # M5 = (A00+A10)B01
    U[0][4] = 1; U[3][4] = 1; V[1][4] = 1
    W[1][4] = 1; W[4][4] = -1
    
    # M6 = (A01-A10)(B10+B11)
    U[1][5] = 1; U[3][5] = -1; V[3][5] = 1; V[4][5] = 1
    W[0][5] = 1; W[4][5] = 1
    
    # M7 = (A00-A01)(B00+B10)
    U[0][6] = 1; U[1][6] = -1; V[0][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[1][6] = 1
    
    # Schoolbook for remaining
    # C02 = A00B02 + A01B12 + A02B22
    for r, (ai, bi) in enumerate([(0, 1), (1, 4), (2, 7)]): # A00B02, A01B12, A02B22
        U[ai][7+r] = 1
        V[bi][7+r] = 1
        W[6][7+r] = 1 # C02 is row 6 (k=2, i=0)
        
    # C12 = A10B02 + A11B12 + A12B22
    for r, (ai, bi) in enumerate([(3, 1), (4, 4), (5, 7)]):
        U[ai][10+r] = 1
        V[bi][10+r] = 1
        W[7][10+r] = 1 # C12 is row 7 (k=2, i=1)
        
    # C20 = A20B00 + A21B10 + A22B20
    for r, (ai, bi) in enumerate([(6, 0), (7, 3), (8, 6)]):
        U[ai][13+r] = 1
        V[bi][13+r] = 1
        W[2][13+r] = 1 # C20 is row 2 (k=0, i=2)
        
    # C21 = A20B01 + A21B11 + A22B21
    for r, (ai, bi) in enumerate([(6, 1), (7, 4), (8, 7)]):
        U[ai][16+r] = 1
        V[bi][16+r] = 1
        W[5][16+r] = 1 # C21 is row 5 (k=1, i=2)
        
    # C22 = A20B02 + A21B12 + A22B22
    for r, (ai, bi) in enumerate([(6, 1), (7, 4), (8, 7)]): # Wait, B indices for C22
        # C22 = A20B02 + A21B12 + A22B22
        # B02 is index 1, B12 is index 4, B22 is index 7
        U[ai][19+r] = 1
        V[bi][19+r] = 1
        W[8][19+r] = 1 # C22 is row 8 (k=2, i=2)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
