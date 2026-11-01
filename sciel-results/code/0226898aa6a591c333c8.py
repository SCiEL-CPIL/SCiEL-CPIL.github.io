"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 14 decomposition based on two Strassen 2x2x2 multiplications
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: A * B1 (B1 is cols 0,1 of B)
    # Strassen M1..M7
    # U rows: 0:a, 1:b, 2:c, 3:d
    # V rows: 0:e, 1:f, 4:g, 5:h (others 0)
    # W rows: 0:C00, 1:C10, 2:C01, 3:C11 (others 0)
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = d(g-c)
    U[3][3] = 1
    V[0][3] = 1; V[4][3] = 1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1
    
    # Block 2: A * B2 (B2 is cols 2,3 of B)
    # V rows: 2:i, 3:j, 6:k, 7:l (others 0)
    # W rows: 4:C02, 5:C12, 6:C03, 7:C13 (others 0)
    
    # M1' = (a+d)(i+l)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # M2' = (c+d)i
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    # M3' = a(j-l)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    # M4' = d(k-c)
    U[3][10] = 1
    V[2][10] = 1; V[6][10] = 1
    W[4][10] = 1; W[5][10] = 1
    
    # M5' = (a+b)l
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    # M6' = (c-a)(i+j)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    # M7' = (b-d)(k+l)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
