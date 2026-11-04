import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 (Strassen) + Rank 16 for 4x4x1 (Schoolbook) = 65
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen Decomposition (Rank 49) ---
    
    # Define 2x2 Strassen decomposition (Rank 7)
    # U22, V22, W22 are 4x7 matrices
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Indices for 2x2 matrix: 00, 01, 10, 11 -> 0, 1, 2, 3
    U22 = np.zeros((4, 7))
    V22 = np.zeros((4, 7))
    W22 = np.zeros((4, 7))
    
    # M1
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1; W22[1, 0] = 1; W22[2, 0] = 1; W22[3, 0] = 1 # C00+=M1, C01+=M1? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W22 maps M_k (2x2) to C (2x2)
    # W22[c_idx, m_idx]
    
    # M1
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1 # C00
    W22[3, 0] = 1 # C11
    
    # M2
    U22[2, 1] = 1; U22[3, 1] = 1
    V22[0, 1] = 1
    W22[2, 1] = 1 # C10
    W22[3, 1] = -1 # C11
    
    # M3
    U22[0, 2] = 1
    V22[1, 2] = 1; V22[3, 2] = -1
    W22[1, 2] = 1 # C01
    W22[3, 2] = 1 # C11
    
    # M4
    U22[3, 3] = 1
    V22[2, 3] = 1; V22[0, 3] = -1
    W22[0, 3] = 1 # C00
    W22[2, 3] = 1 # C10
    
    # M5
    U22[0, 4] = 1; U22[1, 4] = 1
    V22[3, 4] = 1
    W22[0, 4] = -1 # C00
    W22[1, 4] = 1 # C01
    
    # M6
    U22[2, 5] = 1; U22[0, 5] = -1
    V22[0, 5] = 1; V22[1, 5] = 1
    W22[3, 5] = 1 # C11
    
    # M7
    U22[1, 6] = 1; U22[3, 6] = -1
    V22[2, 6] = 1; V22[3, 6] = 1
    W22[0, 6] = 1 # C00
    
    # Now construct 4x4x4 decomposition
    # 7 terms, each rank 7. Total 49.
    # For each term k, we need L_A, L_B, L_C (16x4 matrices)
    
    # A blocks: A00 (0,1,4,5), A01 (2,3,6,7), A10 (8,9,12,13), A11 (10,11,14,15)
    # B blocks: B00 (0,1,5,6), B01 (2,3,7,8), B10 (10,11,15,16), B11 (12,13,17,18)
    # Note: B is 4x4 here. Indices j*4+k.
    # B00: j=0,1 k=0,1 -> 0,1,4,5
    # B01: j=0,1 k=2,3 -> 2,3,6,7
    # B10: j=2,3 k=0,1 -> 8,9,12,13
    # B11: j=2,3 k=2,3 -> 10,11,14,15
    
    # C blocks: C00 (0,1,4,5), C01 (2,3,6,7), C10 (8,9,12,13), C11 (10,11,14,15)
    
    L_A = [np.zeros((4, 16)) for _ in range(7)]
    L_B = [np.zeros((4, 16)) for _ in range(7)]
    L_C = [np.zeros((16, 4)) for _ in range(7)]
    
    # Helper to add block
    def add_block(L, block_idx, coeff, target_vec_idx):
        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
        # target_vec_idx: 0..3 (element in 2x2 vec)
        base = block_idx * 4
        for e in range(4):
            L[target_vec_idx, base + e] = coeff

    # M1: A00+A11, B00+B11
    add_block(L_A[0], 0, 1, 0); add_block(L_A[0], 3, 1, 0)
    add_block(L_A[0], 0, 1, 1); add_block(L_A[0], 3, 1, 1)
    add_block(L_A[0], 0, 1, 2); add_block(L_A[0], 3, 1, 2)
    add_block(L_A[0], 0, 1, 3); add_block(L_A[0], 3, 1, 3)
    
    add_block(L_B[0], 0, 1, 0); add_block(L_B[0], 3, 1, 0)
    add_block(L_B[0], 0, 1, 1); add_block(L_B[0], 3, 1, 1)
    add_block(L_B[0], 0, 1, 2); add_block(L_B[0], 3, 1, 2)
    add_block(L_B[0], 0, 1, 3); add_block(L_B[0], 3, 1, 3)
    
    # C00 += M1, C11 += M1
    L_C[0][0, 0] = 1; L_C[0][1, 1] = 1; L_C[0][4, 2] = 1; L_C[0][5, 3] = 1
    L_C[0][10, 0] = 1; L_C[0][11, 1] = 1; L_C[0][14, 2] = 1; L_C[0][15, 3] = 1

    # M2: A10+A11, B00
    add_block(L_A[1], 2, 1, 0); add_block(L_A[1], 3, 1, 0)
    add_block(L_A[1], 2, 1, 1); add_block(L_A[1], 3, 1, 1)
    add_block(L_A[1], 2, 1, 2); add_block(L_A[1], 3, 1, 2)
    add_block(L_A[1], 2, 1, 3); add_block(L_A[1], 3, 1, 3)
    
    add_block(L_B[1], 0, 1, 0); add_block(L_B[1], 0, 1, 1)
    add_block(L_B[1], 0, 1, 2); add_block(L_B[1], 0, 1, 3)
    
    # C10 += M2, C11 -= M2
    L_C[1][8, 0] = 1; L_C[1][9, 1] = 1; L_C[1][12, 2] = 1; L_C[1][13, 3] = 1
    L_C[1][10, 0] = -1; L_C[1][11, 1] = -1; L_C[1][14, 2] = -1; L_C[1][15, 3] = -1

    # M3: A00, B01-B11
    add_block(L_A[2], 0, 1, 0); add_block(L_A[2], 0, 1, 1)
    add_block(L_A[2], 0, 1, 2); add_block(L_A[2], 0, 1, 3)
    
    add_block(L_B[2], 1, 1, 0); add_block(L_B[2], 3, -1, 0)
    add_block(L_B[2], 1, 1, 1); add_block(L_B[2], 3, -1, 1)
    add_block(L_B[2], 1, 1, 2); add_block(L_B[2], 3, -1, 2)
    add_block(L_B[2], 1, 1, 3); add_block(L_B[2], 3, -1, 3)
    
    # C01 += M3, C11 += M3
    L_C[2][2, 0] = 1; L_C[2][3, 1] = 1; L_C[2][6, 2] = 1; L_C[2][7, 3] = 1
    L_C[2][10, 0] = 1; L_C[2][11, 1] = 1; L_C[2][14, 2] = 1; L_C[2][15, 3] = 1

    # M4: A11, B10-B00
    add_block(L_A[3], 3, 1, 0); add_block(L_A[3], 3, 1, 1)
    add_block(L_A[3], 3, 1, 2); add_block(L_A[3], 3, 1, 3)
    
    add_block(L_B[3], 2, 1, 0); add_block(L_B[3], 0, -1, 0)
    add_block(L_B[3], 2, 1, 1); add_block(L_B[3], 0, -1, 1)
    add_block(L_B[3], 2, 1, 2); add_block(L_B[3], 0, -1, 2)
    add_block(L_B[3], 2, 1, 3); add_block(L_B[3], 0, -1, 3)
    
    # C00 += M4, C10 += M4
    L_C[3][0, 0] = 1; L_C[3][1, 1] = 1; L_C[3][4, 2] = 1; L_C[3][5, 3] = 1
    L_C[3][8, 0] = 1; L_C[3][9, 1] = 1; L_C[3][12, 2] = 1; L_C[3][13, 3] = 1

    # M5: A00+A01, B11
    add_block(L_A[4], 0, 1, 0); add_block(L_A[4], 1, 1, 0)
    add_block(L_A[4], 0, 1, 1); add_block(L_A[4], 1, 1, 1)
    add_block(L_A[4], 0, 1, 2); add_block(L_A[4], 1, 1, 2)
    add_block(L_A[4], 0, 1, 3); add_block(L_A[4], 1, 1, 3)
    
    add_block(L_B[4], 3, 1, 0); add_block(L_B[4], 3, 1, 1)
    add_block(L_B[4], 3, 1, 2); add_block(L_B[4], 3, 1, 3)
    
    # C00 -= M5, C01 += M5
    L_C[4][0, 0] = -1; L_C[4][1, 1] = -1; L_C[4][4, 2] = -1; L_C[4][5, 3] = -1
    L_C[4][2, 0] = 1; L_C[4][3, 1] = 1; L_C[4][6, 2] = 1; L_C[4][7, 3] = 1

    # M6: A10-A00, B00+B01
    add_block(L_A[5], 2, 1, 0); add_block(L_A[5], 0, -1, 0)
    add_block(L_A[5], 2, 1, 1); add_block(L_A[5], 0, -1, 1)
    add_block(L_A[5], 2, 1, 2); add_block(L_A[5], 0, -1, 2)
    add_block(L_A[5], 2, 1, 3); add_block(L_A[5], 0, -1, 3)
    
    add_block(L_B[5], 0, 1, 0); add_block(L_B[5], 1, 1, 0)
    add_block(L_B[5], 0, 1, 1); add_block(L_B[5], 1, 1, 1)
    add_block(L_B[5], 0, 1, 2); add_block(L_B[5], 1, 1, 2)
    add_block(L_B[5], 0, 1, 3); add_block(L_B[5], 1, 1, 3)
    
    # C11 += M6
    L_C[5][10, 0] = 1; L_C[5][11, 1] = 1; L_C[5][14, 2] = 1; L_C[5][15, 3] = 1

    # M7: A01-A11, B10+B11
    add_block(L_A[6], 1, 1, 0); add_block(L_A[6], 3, -1, 0)
    add_block(L_A[6], 1, 1, 1); add_block(L_A[6], 3, -1, 1)
    add_block(L_A[6], 1, 1, 2); add_block(L_A[6], 3, -1, 2)
    add_block(L_A[6], 1, 1, 3); add_block(L_A[6], 3, -1, 3)
    
    add_block(L_B[6], 2, 1, 0); add_block(L_B[6], 3, 1, 0)
    add_block(L_B[6], 2, 1, 1); add_block(L_B[6], 3, 1, 1)
    add_block(L_B[6], 2, 1, 2); add_block(L_B[6], 3, 1, 2)
    add_block(L_B[6], 2, 1, 3); add_block(L_B[6], 3, 1, 3)
    
    # C00 += M7
    L_C[6][0, 0] = 1; L_C[6][1, 1] = 1; L_C[6][4, 2] = 1; L_C[6][5, 3] = 1

    # Fill U, V, W for the first 49 ranks
    for k in range(7):
        for r in range(7):
            col = k * 7 + r
            # U
            for i in range(4):
                for j in range(16):
                    U[i*4+j][col] += U22[i, r] * L_A[k][j, r] # Wait, L_A is 4x16. U22 is 4x7.
                    # Correct: U_vec = L_A * U22?
                    # U22[i, r] is coeff for A_elem_i in M_k.
                    # L_A[k][i, j] is coeff for A_elem_j in A_elem_i of M_k.
                    # So U[j, col] = sum_i U22[i, r] * L_A[k][i, j]
            # Let's do matrix multiplication
            U_block = np.dot(L_A[k], U22[:, r])
            V_block = np.dot(L_B[k], V22[:, r])
            W_block = np.dot(L_C[k], W22[:, r])
            
            for idx in range(16):
                U[idx][col] = U_block[idx]
                V[idx][col] = V_block[idx] # V is 16x? No, V is 20x. For first 4 cols, V is 16x.
                W[idx][col] = W_block[idx]
                
            # Note: V is m*p x rank = 20 x rank.
            # For the first 4 columns of B, the indices are 0..15.
            # L_B is 4x16, so V_block is 16x1.
            # We only fill V[0..15, col]. V[16..19, col] remains 0.

    # --- Part 2: 4x4x1 Schoolbook Decomposition (Rank 16) ---
    # C_i4 = sum_j A_ij B_j4
    for i in range(4):
        for j in range(4):
            col = 49 + i * 4 + j
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1 # k=4, so index 4*4+i = 16+i

    return {"U": U, "V": V, "W": W}
