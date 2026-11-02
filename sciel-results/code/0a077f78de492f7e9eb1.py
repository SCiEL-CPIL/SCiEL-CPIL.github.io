"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U rows: i*m + j
    # 0: a00, 1: a01, 2: a02
    # 3: a10, 4: a11, 5: a12
    # V rows: j*p + k
    # 0: b00, 1: b01, 2: b02
    # 3: b10, 4: b11, 5: b12
    # 6: b20, 7: b21, 8: b22
    # W rows: k*n + i
    # 0: c00, 1: c10
    # 2: c01, 3: c11
    # 4: c02, 5: c12

    # Strassen for A_left (cols 0,1) * B_top_left (rows 0,1, cols 0,1)
    # m1
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1
    
    # m2
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # m3
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # m4
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # m5
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # m6
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # m7
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1

    # Schoolbook for A_left * B_top_right (col 2 of top)
    # m8: a00 * b02 -> c02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # m9: a01 * b12 -> c02
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # m10: a10 * b02 -> c12
    U[3][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # m11: a11 * b12 -> c12
    U[4][10] = 1
    V[5][10] = 1
    W[5][10] = 1

    # Schoolbook for A_right (col 2) * B_bottom (row 2)
    # m12: a02 * b20 -> c00
    U[2][11] = 1
    V[6][11] = 1
    W[0][11] = 1
    
    # m13: a12 * b20 -> c10
    U[5][12] = 1
    V[6][12] = 1
    W[1][12] = 1
    
    # m14: a02 * b21 -> c01
    U[2][13] = 1
    V[7][13] = 1
    W[2][13] = 1
    
    # m15: a12 * b21 -> c11
    U[5][14] = 1
    V[7][14] = 1
    W[3][14] = 1
    
    # m16: a02 * b22 -> c02
    U[2][15] = 1
    V[8][15] = 1
    W[4][15] = 1
    
    # m17: a12 * b22 -> c12
    U[5][16] = 1
    V[8][16] = 1
    W[5][16] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
