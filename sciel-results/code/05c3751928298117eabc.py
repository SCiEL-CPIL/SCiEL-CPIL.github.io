"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    # We construct a rank-25 decomposition for 2x2x7 matrix multiplication.
    # This is achieved by splitting the output dimension (7) into three blocks:
    #   Block 1: 2x2x2 (rank 7, Strassen)
    #   Block 2: 2x2x2 (rank 7, Strassen)
    #   Block 3: 2x2x3 (rank 11, Strassen for first 2 cols + Schoolbook for 3rd)
    # Total rank: 7 + 7 + 11 = 25.
    
    n, m, p = 2, 2, 7
    rank = 25
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 4 x 25
    V = [[0] * rank for _ in range(m * p)]  # 14 x 25
    W = [[0] * rank for _ in range(p * n)]  # 14 x 25
    
    # --- Block 1: k = 0, 1 (indices 0, 1 in output) ---
    # Strassen for 2x2x2
    # A = [[a00, a01], [a10, a11]]
    # B1 = [[b00, b01], [b10, b11]] (first two columns of B)
    # C1 = A * B1
    #
    # Strassen formulas:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Column indices for Block 1: 0..6
    # U rows: 0=a00, 1=a01, 2=a10, 3=a11
    # V rows: 0=b00, 1=b01, 2=b10, 3=b11 (for k=0,1)
    # W rows: 0=c00, 1=c01, 2=c10, 3=c11 (for k=0,1)
    
    # M1: U: a00=1, a11=1; V: b00=1, b11=1; W: c00=1, c11=1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2: U: a10=1, a11=1; V: b00=1; W: c10=1
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    
    # M3: U: a00=1; V: b01=1, b11=-1; W: c01=1
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1
    
    # M4: U: a11=1; V: b10=1, b00=-1; W: c00=1, c10=1
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5: U: a00=1, a01=1; V: b11=1; W: c00=-1, c01=1
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # M6: U: a10=1, a00=-1; V: b00=1, b01=1; W: c11=1
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7: U: a01=1, a11=-1; V: b10=1, b11=1; W: c00=1
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    # --- Block 2: k = 2, 3 (indices 2, 3 in output) ---
    # Same Strassen decomposition, shifted by 7 in column index, 4 in V and W row index.
    # V rows for k=2,3: 4=b00, 5=b01, 6=b10, 7=b11
    # W rows for k=2,3: 4=c00, 5=c01, 6=c10, 7=c11
    
    # M1
    U[0][7] += 1; U[3][7] += 1
    V[4][7] += 1; V[7][7] += 1
    W[4][7] += 1; W[7][7] += 1
    
    # M2
    U[2][8] += 1; U[3][8] += 1
    V[4][8] += 1
    W[6][8] += 1
    
    # M3
    U[0][9] += 1
    V[5][9] += 1; V[7][9] -= 1
    W[5][9] += 1
    
    # M4
    U[3][10] += 1
    V[6][10] += 1; V[4][10] -= 1
    W[4][10] += 1; W[6][10] += 1
    
    # M5
    U[0][11] += 1; U[1][11] += 1
    V[7][11] += 1
    W[4][11] -= 1; W[5][11] += 1
    
    # M6
    U[2][12] += 1; U[0][12] -= 1
    V[4][12] += 1; V[5][12] += 1
    W[7][12] += 1
    
    # M7
    U[1][13] += 1; U[3][13] -= 1
    V[6][13] += 1; V[7][13] += 1
    W[4][13] += 1
    
    # --- Block 3: k = 4, 5, 6 (indices 4, 5, 6 in output) ---
    # Columns 14..24 (11 columns)
    # V rows for k=4,5,6: 8=b00, 9=b01, 10=b10, 11=b11 (k=4,5); 12=b00, 13=b01, 14=b10, 15=b11 (k=6)
    # Wait, V has 14 rows. m*p = 2*7 = 14.
    # V row index: j*p + k.
    # j=0 (a00, a01 multiply), j=1 (a10, a11 multiply)
    # k=4: V[0*7+4]=V[4], V[1*7+4]=V[11]
    # k=5: V[0*7+5]=V[5], V[1*7+5]=V[12]
    # k=6: V[0*7+6]=V[6], V[1*7+6]=V[13]
    
    # Let's re-index V correctly.
    # V[row][col]
    # row = j*p + k
    # For Block 3 (k=4,5,6):
    # k=4: j=0 -> V[4], j=1 -> V[11]
    # k=5: j=0 -> V[5], j=1 -> V[12]
    # k=6: j=0 -> V[6], j=1 -> V[13]
    
    # W rows for k=4,5,6:
    # W[row][col]
    # row = k*n + i
    # k=4: i=0 -> W[8], i=1 -> W[9]
    # k=5: i=0 -> W[10], i=1 -> W[11]
    # k=6: i=0 -> W[12], i=1 -> W[13]
    
    # Part 3a: Strassen for k=4,5 (first two columns of Block 3)
    # Same pattern as Block 1 and 2, but with different V and W row indices.
    # Columns 14..20
    
    # M1
    U[0][14] += 1; U[3][14] += 1
    V[4][14] += 1; V[11][14] += 1
    W[8][14] += 1; W[11][14] += 1
    
    # M2
    U[2][15] += 1; U[3][15] += 1
    V[4][15] += 1
    W[10][15] += 1
    
    # M3
    U[0][16] += 1
    V[5][16] += 1; V[11][16] -= 1
    W[9][16] += 1
    
    # M4
    U[3][17] += 1
    V[6][17] += 1; V[4][17] -= 1
    W[8][17] += 1; W[10][17] += 1
    
    # M5
    U[0][18] += 1; U[1][18] += 1
    V[11][18] += 1
    W[8][18] -= 1; W[9][18] += 1
    
    # M6
    U[2][19] += 1; U[0][19] -= 1
    V[4][19] += 1; V[5][19] += 1
    W[11][19] += 1
    
    # M7
    U[1][20] += 1; U[3][20] -= 1
    V[6][20] += 1; V[11][20] += 1
    W[8][20] += 1
    
    # Part 3b: Schoolbook for k=6 (last column of Block 3)
    # c06 = a00*b06 + a01*b16
    # c16 = a10*b06 + a11*b16
    # Columns 21..24
    
    # a00*b06
    U[0][21] += 1
    V[6][21] += 1
    W[12][21] += 1
    
    # a01*b16
    U[1][22] += 1
    V[13][22] += 1
    W[12][22] += 1
    
    # a10*b06
    U[2][23] += 1
    V[6][23] += 1
    W[13][23] += 1
    
    # a11*b16
    U[3][24] += 1
    V[13][24] += 1
    W[13][24] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
