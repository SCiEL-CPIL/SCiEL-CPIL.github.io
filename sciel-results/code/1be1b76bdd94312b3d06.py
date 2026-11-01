"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 14 decomposition based on Strassen's algorithm applied to two 2x2 blocks
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # Indices: 0: a00, 1: a01, 2: a10, 3: a11
    # Forms:
    # 0: a00 + a11
    # 1: a10 + a11
    # 2: a00
    # 3: a11
    # 4: a00 + a01
    # 5: a10 - a00
    # 6: a01 - a11
    
    # Set U for first 7 terms (B1 part)
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    # Set U for next 7 terms (B2 part) - same forms
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    # Strassen linear forms for B (2x2 blocks)
    # B1 indices in V: 0: b00, 1: b01, 4: b10, 5: b11
    # B2 indices in V: 2: b02, 3: b03, 6: b12, 7: b13
    
    # Set V for first 7 terms (B1)
    # 0: b00 + b11
    V[0][0] = 1; V[5][0] = 1
    # 1: b00
    V[0][1] = 1
    # 2: b01 - b11
    V[1][2] = 1; V[5][2] = -1
    # 3: b10 - b00
    V[4][3] = 1; V[0][3] = -1
    # 4: b11
    V[5][4] = 1
    # 5: b00 + b01
    V[0][5] = 1; V[1][5] = 1
    # 6: b10 + b11
    V[4][6] = 1; V[5][6] = 1
    
    # Set V for next 7 terms (B2)
    # 7: b02 + b13
    V[2][7] = 1; V[7][7] = 1
    # 8: b02
    V[2][8] = 1
    # 9: b03 - b13
    V[3][9] = 1; V[7][9] = -1
    # 10: b12 - b02
    V[6][10] = 1; V[2][10] = -1
    # 11: b13
    V[7][11] = 1
    # 12: b02 + b03
    V[2][12] = 1; V[3][12] = 1
    # 13: b12 + b13
    V[6][13] = 1; V[7][13] = 1
    
    # Output coefficients W
    # W indices: k*n + i
    # C1 (cols 0,1):
    # C00 (W[0]): M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C10 (W[1]): M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C01 (W[2]): M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C11 (W[3]): M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # C2 (cols 2,3):
    # C02 (W[4]): M'1 + M'4 - M'5 + M'7
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    # C12 (W[5]): M'2 + M'4
    W[5][8] = 1; W[5][10] = 1
    # C03 (W[6]): M'3 + M'5
    W[6][9] = 1; W[6][11] = 1
    # C13 (W[7]): M'1 - M'2 + M'3 + M'6
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
