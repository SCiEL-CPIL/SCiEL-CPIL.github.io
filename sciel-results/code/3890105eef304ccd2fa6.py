"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 for A[0:2, 0:2] * B[0:2, 0:2]
    # U indices: A00=0, A01=1, A10=3, A11=4
    # V indices: B00=0, B01=1, B10=3, B11=4
    # W indices: C00=0, C10=1, C01=2, C11=3
    
    # M1 = A00(B00+B11)
    U[0][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[2][0] = 1; W[3][0] = 1
    
    # M2 = (A00+A10)B11
    U[0][1] = 1; U[3][1] = 1
    V[4][1] = 1
    W[0][1] = -1; W[2][1] = 1
    
    # M3 = (A01+A11)B00
    U[1][2] = 1; U[4][2] = 1
    V[0][2] = 1
    W[1][2] = 1; W[3][2] = -1
    
    # M4 = A11(B01+B10)
    U[4][3] = 1
    V[1][3] = 1; V[3][3] = 1
    W[0][3] = 1; W[1][3] = 1; W[3][3] = 1
    
    # M5 = (A00+A11)(B00+B11)
    U[0][4] = 1; U[4][4] = 1
    V[0][4] = 1; V[4][4] = 1
    W[0][4] = 1; W[2][4] = -1; W[3][4] = 1
    
    # M6 = (A00-A10)(B10+B11)
    U[0][5] = 1; U[3][5] = -1
    V[3][5] = 1; V[4][5] = 1
    W[0][5] = 1; W[1][5] = -1
    
    # M7 = (A01+A11)(B00-B10)
    U[1][6] = 1; U[4][6] = 1
    V[0][6] = 1; V[3][6] = -1
    W[2][6] = 1; W[1][6] = -1

    # Schoolbook 2x2x1 for A[0:2, 0:2] * B[0:2, 2:3]
    # U indices: A00=0, A01=1, A10=3, A11=4
    # V indices: B02=2, B12=5
    # W indices: C02=4, C12=5
    
    # M8 = A00 B02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # M9 = A01 B12
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # M10 = A10 B02
    U[3][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # M11 = A11 B12
    U[4][10] = 1
    V[5][10] = 1
    W[5][10] = 1

    # Schoolbook 2x1x3 for A[0:2, 2:3] * B[2:3, 0:3]
    # U indices: A02=2, A12=5
    # V indices: B20=6, B21=7, B22=8
    # W indices: C00=0, C10=1, C01=2, C11=3, C02=4, C12=5
    
    # M12 = A02 B20
    U[2][11] = 1
    V[6][11] = 1
    W[0][11] = 1
    
    # M13 = A02 B21
    U[2][12] = 1
    V[7][12] = 1
    W[2][12] = 1
    
    # M14 = A02 B22
    U[2][13] = 1
    V[8][13] = 1
    W[4][13] = 1
    
    # M15 = A12 B20
    U[5][14] = 1
    V[6][14] = 1
    W[1][14] = 1
    
    # M16 = A12 B21
    U[5][15] = 1
    V[7][15] = 1
    W[3][15] = 1
    
    # M17 = A12 B22
    U[5][16] = 1
    V[8][16] = 1
    W[5][16] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
