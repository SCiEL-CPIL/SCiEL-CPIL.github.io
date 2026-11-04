def construct():
    n, m, p = 4, 4, 9
    
    # We'll use Strassen for 4x4 x 4x4 (rank 49) twice, and schoolbook for 4x4 x 4x1 (rank 16)
    # Total rank: 49 + 49 + 16 = 114
    
    # First, let's build the Strassen 2x2 x 2x2 decomposition (rank 7)
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # m1 = (a+d)(e+h)
    # a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
    # Contributes to C[0,0] and C[1,1]
    U2[0][0] = 1  # a
    U2[3][0] = 1  # d
    V2[0][0] = 1  # e
    V2[3][0] = 1  # h
    W2[0][0] = 1  # C[0,0]
    W2[3][0] = 1  # C[1,1]
    
    # m2 = (c+d)e
    # c=A[1,0], d=A[1,1], e=B[0,0]
    # Contributes to C[1,0] (+1) and C[1,1] (-1)
    U2[2][1] = 1  # c
    U2[3][1] = 1  # d
    V2[0][1] = 1  # e
    W2[2][1] = 1  # C[1,0]
    W2[3][1] = -1 # C[1,1]
    
    # m3 = a(f-h)
    # a=A[0,0], f=B[0,1], h=B[1,1]
    # Contributes to C[0,0] (+1) and C[0,1] (+1)
    U2[0][2] = 1  # a
    V2[1][2] = 1  # f
    V2[3][2] = -1 # h
    W2[0][2] = 1  # C[0,0]
    W2[1][2] = 1  # C[0,1]
    
    # m4 = d(g-e)
    # d=A[1,1], g=B[1,0], e=B[0,0]
    # Contributes to C[0,0] (+1) and C[1,0] (+1)
    U2[3][3] = 1  # d
    V2[2][3] = 1  # g
    V2[0][3] = -1 # e
    W2[0][3] = 1  # C[0,0]
    W2[2][3] = 1  # C[1,0]
    
    # m5 = (a+b)h
    # a=A[0,0], b=A[0,1], h=B[1,1]
    # Contributes to C[0,0] (-1) and C[0,1] (+1)
    U2[0][4] = 1  # a
    U2[1][4] = 1  # b
    V2[3][4] = 1  # h
    W2[0][4] = -1 # C[0,0]
    W2[1][4] = 1  # C[0,1]
    
    # m6 = (c-a)(e+f)
    # c=A[1,0], a=A[0,0], e=B[0,0], f=B[0,1]
    # Contributes to C[1,1] (+1)
    U2[2][5] = 1  # c
    U2[0][5] = -1 # a
    V2[0][5] = 1  # e
    V2[1][5] = 1  # f
    W2[3][5] = 1  # C[1,1]
    
    # m7 = (b-d)(g+h)
    # b=A[0,1], d=A[1,1], g=B[1,0], h=B[1,1]
    # Contributes to C[0,0] (+1) and C[1,1] (+1)
    U2[1][6] = 1  # b
    U2[3][6] = -1 # d
    V2[2][6] = 1  # g
    V2[3][6] = 1  # h
    W2[0][6] = 1  # C[0,0]
    W2[3][6] = 1  # C[1,1]
    
    # Verify the 2x2 decomposition
    # For all (i,j,k) in {0,1}^3, sum_c U2[i*2+j,c] * V2[j*2+k,c] * W2[k*2+i,c] should be 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c]
                assert s == 1, f"2x2 verification failed at ({i},{j},{k}): {s}"
    
    # Now build the 4x4 x 4x4 decomposition using block Strassen
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    
    # Block-level Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mj is a 2x2 x 2x2 product, decomposed with rank 7
    # Total rank: 7*7 = 49
    
    # U4: (16, 49), V4: (16, 49), W4: (16, 49)
    # U4[i*4+j, c], V4[j*4+k, c], W4[k*4+i, c]
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # For each block-level term Mj, and each 2x2 Strassen term c2,
    # we create a rank-1 term in the 4x4 decomposition
    
    # Block indices: 0=A00/B00/C00, 1=A01/B01/C01, 2=A10/B10/C10, 3=A11/B11/C11
    # Block (bi, bj) corresponds to i in [2*bi, 2*bi+1], j in [2*bj, 2*bj+1]
    
    # M1 = (A00+A11)(B00+B11)
    # A factors: A00 (block 0) + A11 (block 3)
    # B factors: B00 (block 0) + B11 (block 3)
    # C factors: C00 (+1), C11 (+1)
    
    # For each c2 in 0..6 (2x2 Strassen term):
    #   For each a_block in [0, 3] (A00, A11):
    #     For each b_block in [0, 3] (B00, B11):
    #       For each c_block in [0, 3] (C00, C11):
    #         U4[...][c] += U2[...][c2] * a_coeff
    #         V4[...][c] += V2[...][c2] * b_coeff
    #         W4[...][c] += W2[...][c2] * c_coeff
    
    # This is getting very complex. Let me use a different approach.
    
    # Actually, let me just verify that the 2x2 Strassen is correct, and then
    # build the 4x4x4 decomposition by "lifting" the 2x2 decomposition.
    
    # The 4x4x4 tensor can be seen as a 2x2 tensor of 2x2x2 tensors.
    # T4[I,J,K] = T2[i,j,k] where I=2*bi+i, J=2*bj+j, K=2*bk+k
    # The block-level Strassen decomposes the 2x2 tensor of 2x2x2 tensors.
    
    # This is equivalent to: T4[I,J,K] = sum_{m=1}^7 sum_{c=1}^7 U4[I,J,m,c] * V4[J,K,m,c] * W4[K,I,m,c]
    # where (m,c) is a combined index for the 49 terms.
    
    # U4[I,J,m,c] = U2_block[I_block,J_block,m] * U2[I,J,c]
    # where U2_block is the block-level U factor.
    
    # Let me define the block-level factors:
    # U_block: (4, 7) where U_block[bi*2+bj, m]
    # V_block: (4, 7) where V_block[bj*2+bk, m]
    # W_block: (4, 7) where W_block[bk*2+bi, m]
    
    U_block = [[0]*7 for _ in range(4)]
    V_block = [[0]*7 for _ in range(4)]
    W_block = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    # U_block: A00 (bi=0,bj=0) + A11 (bi=1,bj=1)
    U_block[0][0] = 1  # A00
    U_block[3][0] = 1  # A11
    # V_block: B00 (bj=0,bk=0) + B11 (bj=1,bk=1)
    V_block[0][0] = 1  # B00
    V_block[3][0] = 1  # B11
    # W_block: C00 (bk=0,bi=0) + C11 (bk=1,bi=1)
    W_block[0][0] = 1  # C00
    W_block[3][0] = 1  # C11
    
    # M2 = (A10+A11)B00
    U_block[2][1] = 1  # A10
    U_block[3][1] = 1  # A11
    V_block[0][1] = 1  # B00
    W_block[2][1] = 1  # C10
    W_block[3][1] = -1 # C11
    
    # M3 = A00(B01-B11)
    U_block[0][2] = 1  # A00
    V_block[1][2] = 1  # B01
    V_block[3][2] = -1 # B11
    W_block[1][2] = 1  # C01
    
    # M4 = A11(B10-B00)
    U_block[3][3] = 1  # A11
    V_block[2][3] = 1  # B10
    V_block[0][3] = -1 # B00
    W_block[0][3] = 1  # C00
    W_block[2][3] = 1  # C10
    
    # M5 = (A00+A01)B11
    U_block[0][4] = 1  # A00
    U_block[1][4] = 1  # A01
    V_block[3][4] = 1  # B11
    W_block[0][4] = -1 # C00
    W_block[1][4] = 1  # C01
    
    # M6 = (A10-A00)(B00+B01)
    U_block[2][5] = 1  # A10
    U_block[0][5] = -1 # A00
    V_block[0][5] = 1  # B00
    V_block[1][5] = 1  # B01
    W_block[3][5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U_block[1][6] = 1  # A01
    U_block[3][6] = -1 # A11
    V_block[2][6] = 1  # B10
    V_block[3][6] = 1  # B11
    W_block[0][6] = 1  # C00
    W_block[3][6] = 1  # C11
    
    # Now build U4, V4, W4
    # U4[I*4+J, m*7+c] = U_block[bi*2+bj, m] * U2[i*2+j, c]
    # where I = 2*bi + i, J = 2*bj + j
    
    for m in range(7):
        for c in range(7):
            col = m * 7 + c
            for bi in range(2):
                for bj in range(2):
                    for i in range(2):
                        for j in range(2):
                            I = 2*bi + i
                            J = 2*bj + j
                            U4[I*4+J][col] += U_block[bi*2+bj][m] * U2[i*2+j][c]
            
            for bj in range(2):
                for bk in range(2):
                    for j in range(2):
                        for k in range(2):
                            J = 2*bj + j
                            K = 2*bk + k
                            V4[J*4+K][col] += V_block[bj*2+bk][m] * V2[j*2+k][c]
            
            for bk in range(2):
                for bi in range(2):
                    for k in range(2):
                        for i in range(2):
                            K = 2*bk + k
                            I = 2*bi + i
                            W4[K*4+I][col] += W_block[bk*2+bi][m] * W2[k*2+i][c]
    
    # Verify the 4x4x4 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U4[i*4+j][c] * V4[j*4+k][c] * W4[k*4+i][c]
                if s != 1:
                    print(f"4x4x4 verification failed at ({i},{j},{k}): {s}")
    
    # Now build the full 4x4 x 4x9 decomposition
    # B = [B1, B2, B3] where B1 is 4x4, B2 is 4x4, B3 is 4x1
    # C = [C1, C2, C3] where C1 = A*B1, C2 = A*B2, C3 = A*B3
    
    # For B1 (columns 0-3): use U4, V4 (first 4 columns of B)
    # For B2 (columns 4-7): use U4, V4 (next 4 columns of B)
    # For B3 (column 8): use schoolbook
    
    rank = 49 + 49 + 16  # 114
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Part 1: A*B1 (columns 0-3 of B, columns 0-3 of C)
    for c in range(49):
        for i in range(4):
            for j in range(4):
                U[i*4+j][c] = U4[i*4+j][c]
            for k in range(4):
                V[j*4+k][c] = V4[j*4+k][c]
            for i2 in range(4):
                W[k*4+i2][c] = W4[k*4+i2][c]
    
    # Part 2: A*B2 (columns 4-7 of B, columns 4-7 of C)
    for c in range(49):
        col = 49 + c
        for i in range(4):
            for j in range(4):
                U[i*4+j][col] = U4[i*4+j][c]
            for k in range(4):
                V[j*4+4+k][col] = V4[j*4+k][c]
            for i2 in range(4):
                W[4+k*4+i2][col] = W4[k*4+i2][c]
    
    # Part 3: A*B3 (column 8 of B, column 8 of C) - schoolbook
    for c in range(16):
        col = 98 + c
        i = c // 4
        j = c % 4
        U[i*4+j][col] = 1
        V[j*4+8][col] = 1
        W[8*4+i][col] = 1
    
    # Verify the full decomposition
    for i in range(4):
        for j in range(4):
            for k in range(9):
                s = 0
                for c in range(rank):
                    s += U[i*4+j][c] * V[j*4+k][c] * W[k*4+i][c]
                if s != 1:
                    print(f"Full verification failed at ({i},{j},{k}): {s}")
    
    return {"U": U, "V": V, "W": W}
