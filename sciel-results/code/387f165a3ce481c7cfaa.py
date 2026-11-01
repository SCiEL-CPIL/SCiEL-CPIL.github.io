"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 (rank 7)
    # P1: A00 * (b11 - b10)
    U[0][0] = 1
    V[4][0] = 1
    V[3][0] = -1
    
    # P2: (A00 + A01) * b10
    U[0][1] = 1
    U[1][1] = 1
    V[3][1] = 1
    
    # P3: (A10 + A11) * b00
    U[2][2] = 1
    U[3][2] = 1
    V[0][2] = 1
    
    # P4: A11 * (b01 - b00)
    U[3][3] = 1
    V[1][3] = 1
    V[0][3] = -1
    
    # P5: (A00 + A11) * (b00 + b11)
    U[0][4] = 1
    U[3][4] = 1
    V[0][4] = 1
    V[4][4] = 1
    
    # P6: (A01 - A10) * (b10 + b11)
    U[1][5] = 1
    U[2][5] = -1
    V[3][5] = 1
    V[4][5] = 1
    
    # P7: (A00 - A10) * (b00 + b01)
    U[0][6] = 1
    U[2][6] = -1
    V[0][6] = 1
    V[1][6] = 1
    
    # Schoolbook part for column 2 (rank 4)
    # Q1: A00 * b02
    U[0][7] = 1
    V[2][7] = 1
    
    # Q2: A01 * b12
    U[1][8] = 1
    V[5][8] = 1
    
    # Q3: A10 * b02
    U[2][9] = 1
    V[2][9] = 1
    
    # Q4: A11 * b12
    U[3][10] = 1
    V[5][10] = 1
    
    # Output reconstruction
    # C00 = P5 + P4 - P2 + P6
    W[0][4] = 1
    W[0][3] = 1
    W[0][1] = -1
    W[0][5] = 1
    
    # C10 = P3 + P7
    W[1][2] = 1
    W[1][6] = 1
    
    # C01 = P1 + P5
    W[2][0] = 1
    W[2][4] = 1
    
    # C11 = P1 + P4 - P7 + P3
    W[3][0] = 1
    W[3][3] = 1
    W[3][6] = -1
    W[3][2] = 1
    
    # C02 = Q1 + Q2
    W[4][7] = 1
    W[4][8] = 1
    
    # C12 = Q3 + Q4
    W[5][9] = 1
    W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
