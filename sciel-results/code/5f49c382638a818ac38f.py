import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # We'll use rank 46: two copies of Bini et al. rank-23 for 3x3x3
    # Split B into B1 (cols 0-2) and B2 (cols 3-5)
    # C = [A*B1, A*B2]
    
    # Bini et al. rank-23 coefficients for 3x3x3
    # U, V, W are 9x23
    # Indices: for 3x3 matrix, index = row*3 + col
    
    U_33 = [[0]*23 for _ in range(9)]
    V_33 = [[0]*23 for _ in range(9)]
    W_33 = [[0]*23 for _ in range(9)]
    
    # r0: a00*b00
    U_33[0][0] = 1; V_33[0][0] = 1; W_33[0][0] = 1
    # r1: a01*b10
    U_33[1][1] = 1; V_33[3][1] = 1; W_33[1][1] = 1
    # r2: a02*b20
    U_33[2][2] = 1; V_33[6][2] = 1; W_33[2][2] = 1
    # r3: a10*b01
    U_33[3][3] = 1; V_33[1][3] = 1; W_33[3][3] = 1
    # r4: a11*b11
    U_33[4][4] = 1; V_33[4][4] = 1; W_33[4][4] = 1
    # r5: a12*b21
    U_33[5][5] = 1; V_33[7][5] = 1; W_33[5][5] = 1
    # r6: a20*b02
    U_33[6][6] = 1; V_33[2][6] = 1; W_33[6][6] = 1
    # r7: a21*b12
    U_33[7][7] = 1; V_33[5][7] = 1; W_33[7][7] = 1
    # r8: a22*b22
    U_33[8][8] = 1; V_33[8][8] = 1; W_33[8][8] = 1
    # r9: (a00+a11)*(b00+b11)
    U_33[0][9] = 1; U_33[4][9] = 1
    V_33[0][9] = 1; V_33[4][9] = 1
    W_33[0][9] = 1; W_33[4][9] = 1
    # r10: (a00+a22)*(b00+b22)
    U_33[0][10] = 1; U_33[8][10] = 1
    V_33[0][10] = 1; V_33[8][10] = 1
    W_33[0][10] = 1; W_33[8][10] = 1
    # r11: (a01+a12)*(b10+b21)
    U_33[1][11] = 1; U_33[5][11] = 1
    V_33[3][11] = 1; V_33[7][11] = 1
    W_33[1][11] = 1; W_33[5][11] = 1
    # r12: (a10+a21)*(b01+b12)
    U_33[3][12] = 1; U_33[7][12] = 1
    V_33[1][12] = 1; V_33[5][12] = 1
    W_33[3][12] = 1; W_33[7][12] = 1
    # r13: (a02+a20)*(b20+b02)
    U_33[2][13] = 1; U_33[6][13] = 1
    V_33[6][13] = 1; V_33[2][13] = 1
    W_33[2][13] = 1; W_33[6][13] = 1
    # r14-r22: I don't have these exactly. Let me use a fallback.
    
    # Since I don't have the full Bini et al. coefficients, I'll use schoolbook for safety
    # but with the structure for 3x3x6
    
    # Actually, let me just use schoolbook rank 54 to ensure correctness
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
