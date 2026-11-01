"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 14 decomposition using Strassen's algorithm for two 2x2 multiplications
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for U: i*m + j
    # Indices for V: j*p + k
    # Indices for W: k*n + i
    
    # Block 1: B cols 0, 1
    # M1: (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3: A00(B10 - B11)
    U[0][2] = 1
    V[4][2] = 1; V[5][2] = -1
    W[3][2] = -1
    
    # M4: A11 B11
    U[3][3] = 1
    V[5][3] = 1
    W[0][3] = -1; W[1][3] = 1
    
    # M5: (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[2][4] = -1; W[3][4] = 1
    
    # M6: (A10 - A00)(B00 + B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[4][5] = 1
    W[0][5] = 1; W[2][5] = 1
    
    # M7: (A01 - A11)(B01 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[5][6] = 1
    W[2][6] = 1
    
    # Block 2: B cols 2, 3
    # M8: (A00 + A11)(B02 + B13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # M9: (A10 + A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1
    
    # M10: A00(B12 - B13)
    U[0][9] = 1
    V[6][9] = 1; V[7][9] = -1
    W[7][9] = -1
    
    # M11: A11 B13
    U[3][10] = 1
    V[7][10] = 1
    W[4][10] = -1; W[5][10] = 1
    
    # M12: (A00 + A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[6][11] = -1; W[7][11] = 1
    
    # M13: (A10 - A00)(B02 + B12)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[6][12] = 1
    W[4][12] = 1; W[6][12] = 1
    
    # M14: (A01 - A11)(B03 + B13)
    U[1][13] = 1; U[3][13] = -1
    V[3][13] = 1; V[7][13] = 1
    W[6][13] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
