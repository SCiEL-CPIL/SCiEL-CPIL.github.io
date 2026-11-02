"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for k=0,1
    # P1 = (a+d)(e+g)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[6][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-f)
    U[3][3] = 1
    V[5][3] = 1; V[1][3] = -1
    W[0][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[6][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[5][6] = 1; V[6][6] = 1
    W[0][6] = -1; W[1][6] = 1
    
    # Schoolbook for k=2
    # a * B[0,2]
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    # b * B[1,2]
    U[1][8] = 1
    V[7][8] = 1
    W[4][8] = 1
    # c * B[0,2]
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    # d * B[1,2]
    U[3][10] = 1
    V[7][10] = 1
    W[5][10] = 1
    
    # Schoolbook for k=3
    # a * B[0,3]
    U[0][11] = 1
    V[3][11] = 1
    W[6][11] = 1
    # b * B[1,3]
    U[1][12] = 1
    V[8][12] = 1
    W[6][12] = 1
    # c * B[0,3]
    U[2][13] = 1
    V[3][13] = 1
    W[7][13] = 1
    # d * B[1,3]
    U[3][14] = 1
    V[8][14] = 1
    W[7][14] = 1
    
    # Schoolbook for k=4
    # a * B[0,4]
    U[0][15] = 1
    V[4][15] = 1
    W[8][15] = 1
    # b * B[1,4]
    U[1][16] = 1
    V[9][16] = 1
    W[8][16] = 1
    # c * B[0,4]
    U[2][17] = 1
    V[4][17] = 1
    W[9][17] = 1
    # d * B[1,4]
    U[3][18] = 1
    V[9][18] = 1
    W[9][18] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
