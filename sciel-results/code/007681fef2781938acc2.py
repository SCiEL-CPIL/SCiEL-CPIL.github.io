"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 25 decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x2x1
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 tensor
    def add_tensor(u_indices, v_indices, w_indices, coeff=1):
        # This is not quite right, we need to add to specific columns
        pass

    # We will just iterate through the 25 terms and set the values
    
    # Indices for U (A matrix 2x2)
    # u[i*m + j]
    # 0: A00, 1: A01, 2: A10, 3: A11
    
    # Indices for V (B matrix 2x7)
    # v[j*p + k]
    # 0: B00, 1: B01, 2: B02, 3: B03, 4: B04, 5: B05, 6: B06
    # 7: B10, 8: B11, 9: B12, 10: B13, 11: B14, 12: B15, 13: B16
    
    # Indices for W (C matrix 2x7, stored as k*n+i)
    # w[k*n + i]
    # k=0: 0: C00, 1: C10
    # k=1: 2: C01, 3: C11
    # k=2: 4: C02, 5: C12
    # k=3: 6: C03, 7: C13
    # k=4: 8: C04, 9: C14
    # k=5: 10: C05, 11: C15
    # k=6: 12: C06, 13: C16
    
    # Strassen Block 1 (k=0,1)
    # M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[7][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] -= 1
    
    # M3
    U[0][2] += 1
    V[1][2] += 1; V[8][2] -= 1
    W[2][2] += 1; W[3][2] += 1
    
    # M4
    U[3][3] += 1
    V[7][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[1][3] += 1
    
    # M5
    U[0][4] += 1; U[1][4] += 1
    V[8][4] += 1
    W[0][4] -= 1; W[2][4] += 1
    
    # M6
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7
    U[1][6] += 1; U[3][6] -= 1
    V[7][6] += 1; V[8][6] += 1
    W[0][6] += 1

    # Strassen Block 2 (k=2,3)
    # M1
    U[0][7] += 1; U[3][7] += 1
    V[2][7] += 1; V[9][7] += 1
    W[4][7] += 1; W[7][7] += 1
    
    # M2
    U[2][8] += 1; U[3][8] += 1
    V[2][8] += 1
    W[5][8] += 1; W[7][8] -= 1
    
    # M3
    U[0][9] += 1
    V[3][9] += 1; V[10][9] -= 1
    W[6][9] += 1; W[7][9] += 1
    
    # M4
    U[3][10] += 1
    V[9][10] += 1; V[2][10] -= 1
    W[4][10] += 1; W[5][10] += 1
    
    # M5
    U[0][11] += 1; U[1][11] += 1
    V[10][11] += 1
    W[4][11] -= 1; W[6][11] += 1
    
    # M6
    U[2][12] += 1; U[0][12] -= 1
    V[2][12] += 1; V[3][12] += 1
    W[7][12] += 1
    
    # M7
    U[1][13] += 1; U[3][13] -= 1
    V[9][13] += 1; V[10][13] += 1
    W[4][13] += 1

    # Strassen Block 3 (k=4,5)
    # M1
    U[0][14] += 1; U[3][14] += 1
    V[4][14] += 1; V[11][14] += 1
    W[8][14] += 1; W[11][14] += 1
    
    # M2
    U[2][15] += 1; U[3][15] += 1
    V[4][15] += 1
    W[9][15] += 1; W[11][15] -= 1
    
    # M3
    U[0][16] += 1
    V[5][16] += 1; V[12][16] -= 1
    W[10][16] += 1; W[11][16] += 1
    
    # M4
    U[3][17] += 1
    V[11][17] += 1; V[4][17] -= 1
    W[8][17] += 1; W[9][17] += 1
    
    # M5
    U[0][18] += 1; U[1][18] += 1
    V[12][18] += 1
    W[8][18] -= 1; W[10][18] += 1
    
    # M6
    U[2][19] += 1; U[0][19] -= 1
    V[4][19] += 1; V[5][19] += 1
    W[11][19] += 1
    
    # M7
    U[1][20] += 1; U[3][20] -= 1
    V[11][20] += 1; V[12][20] += 1
    W[8][20] += 1

    # Schoolbook Block 4 (k=6)
    # C06 = A00*B06 + A01*B16
    U[0][21] += 1
    V[6][21] += 1
    W[12][21] += 1
    
    U[1][22] += 1
    V[13][22] += 1
    W[12][22] += 1
    
    # C16 = A10*B06 + A11*B16
    U[2][23] += 1
    V[6][23] += 1
    W[13][23] += 1
    
    U[3][24] += 1
    V[13][24] += 1
    W[13][24] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
