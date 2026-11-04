"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x2x4 multiplication using two Strassen blocks
    # A indices: i in {0,1}, j in {0,1}
    # U index: i*m + j
    
    # A combinations
    # r=0: A00 + A11
    U[0][0] = 1; U[4][0] = 1
    # r=1: A10 + A11
    U[3][1] = 1; U[4][1] = 1
    # r=2: A00
    U[0][2] = 1
    # r=3: A11
    U[4][3] = 1
    # r=4: A00 + A01
    U[0][4] = 1; U[1][4] = 1
    # r=5: A10 - A00
    U[3][5] = 1; U[0][5] = -1
    # r=6: A01 - A11
    U[1][6] = 1; U[4][6] = -1
    
    # B combinations
    # B indices: j in {0,1}, k in {0,1,2,3}
    # V index: j*p + k
    
    # Block 1 (k=0,1)
    # r=0: B00 + B11
    V[0][0] = 1; V[5][0] = 1
    # r=1: B00
    V[0][1] = 1
    # r=2: B01 - B11
    V[1][2] = 1; V[5][2] = -1
    # r=3: B10 - B00
    V[4][3] = 1; V[0][3] = -1
    # r=4: B11
    V[5][4] = 1
    # r=5: B00 + B01
    V[0][5] = 1; V[1][5] = 1
    # r=6: B10 + B11
    V[4][6] = 1; V[5][6] = 1
    
    # Block 2 (k=2,3)
    # r=7: B02 + B13
    V[2][7] = 1; V[7][7] = 1
    # r=8: B02
    V[2][8] = 1
    # r=9: B03 - B13
    V[3][9] = 1; V[7][9] = -1
    # r=10: B12 - B02
    V[6][10] = 1; V[2][10] = -1
    # r=11: B13
    V[7][11] = 1
    # r=12: B02 + B03
    V[2][12] = 1; V[3][12] = 1
    # r=13: B12 + B13
    V[6][13] = 1; V[7][13] = 1
    
    # W combinations
    # W index: k*n + i
    
    # Block 1 (C cols 0,1)
    # C[0,0] = P1 - P4 + P5 - P7 (idx 0)
    W[0][0] = 1; W[0][3] = -1; W[0][4] = 1; W[0][6] = -1
    # C[0,1] = P3 + P7 (idx 2)
    W[2][2] = 1; W[2][6] = 1
    # C[1,0] = P2 + P4 (idx 1)
    W[1][1] = 1; W[1][3] = 1
    # C[1,1] = P1 - P2 + P3 + P6 (idx 3)
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Block 2 (C cols 2,3)
    # C[0,2] = P8 - P11 + P12 - P14 (idx 4)
    W[4][7] = 1; W[4][10] = -1; W[4][11] = 1; W[4][13] = -1
    # C[0,3] = P10 + P14 (idx 6)
    W[6][9] = 1; W[6][13] = 1
    # C[1,2] = P9 + P11 (idx 5)
    W[5][8] = 1; W[5][10] = 1
    # C[1,3] = P8 - P9 + P10 + P13 (idx 7)
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    # Part 2: 2x1x4 multiplication (j=2)
    # r=14
    # U[i*3+2, 14] = 1
    U[2][14] = 1
    U[5][14] = 1
    # V[2*4+k, 14] = 1
    V[8][14] = 1
    V[9][14] = 1
    V[10][14] = 1
    V[11][14] = 1
    # W[k*2+i, 14] = 1
    for idx in range(8):
        W[idx][14] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
