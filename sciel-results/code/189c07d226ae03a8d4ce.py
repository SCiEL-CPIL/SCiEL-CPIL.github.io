"""Matrix multiplication 3x4 by 4x4 using block decomposition with Strassen."""
DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # For each 3x2 x 2x2 block, we use:
    # - Strassen for the 2x2x2 part (rows 0-1 of the 3-row input): rank 7
    # - Standard for the 1x2x2 part (row 2 of the 3-row input): rank 4
    # Total rank per block: 11
    
    rank = 44  # 4 blocks * 11
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44
    
    # Strassen decomposition for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Standard decomposition for 1x2x2:
    # C0 = A0*B00
    # C1 = A0*B01
    # C2 = A1*B00
    # C3 = A1*B01
    
    # Define the 11 ranks for a 3x2x2 block
    # U_local: 6 rows (i*2+j for i in 0..2, j in 0..1)
    # V_local: 4 rows (j*2+k for j in 0..1, k in 0..1)
    # W_local: 6 rows (k*3+i for k in 0..1, i in 0..2)
    
    U_local = [[0]*11 for _ in range(6)]
    V_local = [[0]*11 for _ in range(4)]
    W_local = [[0]*11 for _ in range(6)]
    
    # Ranks 0-6: Strassen for 2x2x2 (uses rows 0-1 of input)
    # M1 = (A00+A11)*(B00+B11)
    U_local[0][0] = 1  # A00
    U_local[3][0] = 1  # A11
    V_local[0][0] = 1  # B00
    V_local[3][0] = 1  # B11
    W_local[0][0] = 1  # C00
    W_local[3][0] = 1  # C11
    
    # M2 = (A10+A11)*B00
    U_local[2][1] = 1  # A10
    U_local[3][1] = 1  # A11
    V_local[0][1] = 1  # B00
    W_local[3][1] = 1  # C10
    
    # M3 = A00*(B01-B11)
    U_local[0][2] = 1  # A00
    V_local[1][2] = 1  # B01
    V_local[3][2] = -1  # -B11
    W_local[1][2] = 1  # C01
    W_local[3][2] = 1  # C11
    
    # M4 = A11*(B10-B00)
    U_local[3][3] = 1  # A11
    V_local[2][3] = 1  # B10
    V_local[0][3] = -1  # -B00
    W_local[0][3] = 1  # C00
    W_local[3][3] = 1  # C10
    
    # M5 = (A00+A10)*B11
    U_local[0][4] = 1  # A00
    U_local[2][4] = 1  # A10
    V_local[3][4] = 1  # B11
    W_local[0][4] = -1  # -C00
    W_local[1][4] = 1  # C01
    
    # M6 = (A11-A00)*(B00+B10)
    U_local[3][5] = 1  # A11
    U_local[0][5] = -1  # -A00
    V_local[0][5] = 1  # B00
    V_local[2][5] = 1  # B10
    W_local[3][5] = 1  # C11
    
    # M7 = (A01-A11)*(B01+B11)
    U_local[1][6] = 1  # A01
    U_local[3][6] = -1  # -A11
    V_local[1][6] = 1  # B01
    V_local[3][6] = 1  # B11
    W_local[0][6] = 1  # C00
    W_local[1][6] = 1  # C01
    
    # Ranks 7-10: Standard for 1x2x2 (uses row 2 of input)
    # C0 = A20*B00
    U_local[4][7] = 1  # A20
    V_local[0][7] = 1  # B00
    W_local[4][7] = 1  # C0 (row 2, col 0)
    
    # C1 = A20*B01
    U_local[4][8] = 1  # A20
    V_local[1][8] = 1  # B01
    W_local[5][8] = 1  # C1 (row 2, col 1)
    
    # C2 = A21*B00
    U_local[5][9] = 1  # A21
    V_local[0][9] = 1  # B00
    W_local[4][9] = 1  # C2 (row 2, col 0)
    
    # C3 = A21*B01
    U_local[5][10] = 1  # A21
    V_local[1][10] = 1  # B01
    W_local[5][10] = 1  # C3 (row 2, col 1)
    
    # Now map the 4 blocks to global indices
    # Block 0: A[0:3,0:2] x B[0:2,0:2] -> C[0:3,0:2]
    #   U row i*2+j -> global row i*4 + j (j in 0..1)
    #   V row j*2+k -> global row j*4 + k (j in 0..1, k in 0..1)
    #   W row k*3+i -> global row k*3 + i (k in 0..1, i in 0..2)
    
    for r in range(11):
        for u_row in range(6):
            if U_local[u_row][r] != 0:
                i = u_row // 2
                j = u_row % 2
                global_u = i * 4 + j
                U[global_u][r] = U_local[u_row][r]
        for v_row in range(4):
            if V_local[v_row][r] != 0:
                j = v_row // 2
                k = v_row % 2
                global_v = j * 4 + k
                V[global_v][r] = V_local[v_row][r]
        for w_row in range(6):
            if W_local[w_row][r] != 0:
                k = w_row // 3
                i = w_row % 3
                global_w = k * 3 + i
                W[global_w][r] = W_local[w_row][r]
    
    # Block 1: A[0:3,2:4] x B[2:4,0:2] -> C[0:3,0:2]
    #   U row i*2+j -> global row i*4 + (j+2) (j in 0..1, so j+2 in 2..3)
    #   V row j*2+k -> global row (j+2)*4 + k (j in 0..1, so j+2 in 2..3)
    #   W row k*3+i -> global row k*3 + i (k in 0..1, i in 0..2)
    
    for r in range(11):
        for u_row in range(6):
            if U_local[u_row][r] != 0:
                i = u_row // 2
                j = u_row % 2
                global_u = i * 4 + (j + 2)
                U[global_u][11 + r] = U_local[u_row][r]
        for v_row in range(4):
            if V_local[v_row][r] != 0:
                j = v_row // 2
                k = v_row % 2
                global_v = (j + 2) * 4 + k
                V[global_v][11 + r] = V_local[v_row][r]
        for w_row in range(6):
            if W_local[w_row][r] != 0:
                k = w_row // 3
                i = w_row % 3
                global_w = k * 3 + i
                W[global_w][11 + r] = W_local[w_row][r]
    
    # Block 2: A[0:3,0:2] x B[0:2,2:4] -> C[0:3,2:4]
    #   U row i*2+j -> global row i*4 + j (j in 0..1)
    #   V row j*2+k -> global row j*4 + (k+2) (k in 0..1, so k+2 in 2..3)
    #   W row k*3+i -> global row (k+2)*3 + i (k in 0..1, so k+2 in 2..3)
    
    for r in range(11):
        for u_row in range(6):
            if U_local[u_row][r] != 0:
                i = u_row // 2
                j = u_row % 2
                global_u = i * 4 + j
                U[global_u][22 + r] = U_local[u_row][r]
        for v_row in range(4):
            if V_local[v_row][r] != 0:
                j = v_row // 2
                k = v_row % 2
                global_v = j * 4 + (k + 2)
                V[global_v][22 + r] = V_local[v_row][r]
        for w_row in range(6):
            if W_local[w_row][r] != 0:
                k = w_row // 3
                i = w_row % 3
                global_w = (k + 2) * 3 + i
                W[global_w][22 + r] = W_local[w_row][r]
    
    # Block 3: A[0:3,2:4] x B[2:4,2:4] -> C[0:3,2:4]
    #   U row i*2+j -> global row i*4 + (j+2)
    #   V row j*2+k -> global row (j+2)*4 + (k+2)
    #   W row k*3+i -> global row (k+2)*3 + i
    
    for r in range(11):
        for u_row in range(6):
            if U_local[u_row][r] != 0:
                i = u_row // 2
                j = u_row % 2
                global_u = i * 4 + (j + 2)
                U[global_u][33 + r] = U_local[u_row][r]
        for v_row in range(4):
            if V_local[v_row][r] != 0:
                j = v_row // 2
                k = v_row % 2
                global_v = (j + 2) * 4 + (k + 2)
                V[global_v][33 + r] = V_local[v_row][r]
        for w_row in range(6):
            if W_local[w_row][r] != 0:
                k = w_row // 3
                i = w_row % 3
                global_w = (k + 2) * 3 + i
                W[global_w][33 + r] = W_local[w_row][r]
    
    return {"U": U, "V": V, "W": W}
