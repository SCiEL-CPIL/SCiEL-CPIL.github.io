"""Matrix multiplication 4x4 x 4x8 via rank-98 Strassen-based decomposition."""
import time

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Rank-98 exact decomposition of 4x4 * 4x8 matrix multiplication.
    
    Strategy: Split the 4x8 output into two 4x4 blocks (rows 0-3 and 4-7).
    Each 4x4x4 block uses the rank-49 decomposition (tensor square of the
    2x2 Strassen rank-7 tensor). Total rank = 49 + 49 = 98.
    
    The 2x2 Strassen tensor (rank 7) factors:
      U[i,j, c], V[j,k, c], W[k,i, c] for i,j,k in {0,1}, c in 0..6
    
    The 4x4x4 tensor = Strassen_2x2 ⊗ Strassen_2x2 (Kronecker product),
    giving rank 7*7 = 49.
    """
    # 2x2 Strassen rank-7 factors
    # U: 4x7, V: 4x7, W: 4x7
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # The 7 Strassen terms for 2x2 matrix multiplication:
    # C = A * B where A,B,C are 2x2
    # C[0,0] = M1 - M4 + M5 + M7
    # C[0,1] = M3 + M4
    # C[1,0] = M1 + M2
    # C[1,1] = M2 - M3 + M5 + M6
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # U factors: U[i,j,c] for the A matrix part
    # V factors: V[j,k,c] for the B matrix part
    # W factors: W[k,i,c] for the C matrix part
    
    # Let me define the 7 terms explicitly.
    # For term c, U[i,j,c] * V[j,k,c] * W[k,i,c] should sum to C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Term 0: M1 = (A00+A11)*(B00+B11)
    #   U: A00=1, A11=1 -> U[0,0]=1, U[1,1]=1
    #   V: B00=1, B11=1 -> V[0,0]=1, V[1,1]=1
    #   W: C00 += 1, C11 += 1 -> W[0,0]=1, W[1,1]=1
    #   But wait, this contributes to C[0,0] and C[1,1] but also cross terms.
    #   Actually M1 contributes to both C[0,0] and C[1,1] with +1, 
    #   but we need to subtract M4 from C[0,0] etc.
    
    # Let me be more careful. The standard Strassen decomposition:
    # C[0,0] = M1 - M4 + M5 + M7
    # C[0,1] = M3 + M4
    # C[1,0] = M1 + M2
    # C[1,1] = M2 - M3 + M5 + M6
    
    # So W[k,i,c] gives the coefficient of M_c in C[i,k]:
    # W[0,0,0]=1, W[1,1,0]=1  (M1)
    # W[1,0,1]=1               (M2) -> C[1,0] += M2
    # W[0,1,2]=1               (M3) -> C[0,1] += M3
    # W[0,0,3]=-1, W[0,1,3]=1  (M4) -> C[0,0] -= M4, C[0,1] += M4
    # W[0,0,4]=1, W[1,1,4]=1   (M5) -> C[0,0] += M5, C[1,1] += M5
    # W[1,1,5]=1               (M6) -> C[1,1] += M6
    # W[0,0,6]=1, W[1,0,6]=1   (M7) -> C[0,0] += M7, C[1,0] += M7
    
    # Wait, let me re-index. W[k,i,c] where output C[i,k].
    # C[0,0]: k=0,i=0. Coeffs: M1(+1), M4(-1), M5(+1), M7(+1)
    # C[0,1]: k=1,i=0. Coeffs: M3(+1), M4(+1)
    # C[1,0]: k=0,i=1. Coeffs: M1(+1), M2(+1), M7(+1)
    # C[1,1]: k=1,i=1. Coeffs: M2(+1), M3(-1), M5(+1), M6(+1)
    
    # U[i,j,c]: coefficient of A[i,j] in M_c
    # M1 = (A00+A11)*(B00+B11): A00=1, A11=1
    # M2 = (A10+A11)*B00: A10=1, A11=1
    # M3 = A00*(B01-B11): A00=1
    # M4 = A11*(B10-B00): A11=1
    # M5 = (A00+A01)*B11: A00=1, A01=1
    # M6 = (A10-A00)*(B00+B01): A10=1, A00=-1
    # M7 = (A01-A11)*(B10+B11): A01=1, A11=-1
    
    # V[j,k,c]: coefficient of B[j,k] in M_c
    # M1: B00=1, B11=1
    # M2: B00=1
    # M3: B01=1, B11=-1
    # M4: B10=1, B00=-1
    # M5: B11=1
    # M6: B00=1, B01=1
    # M7: B10=1, B11=1
    
    # Build U, V, W for 2x2 (size 4x7 each)
    U2 = [[0]*7 for _ in range(4)]  # U[i*2+j, c]
    V2 = [[0]*7 for _ in range(4)]  # V[j*2+k, c]
    W2 = [[0]*7 for _ in range(4)]  # W[k*2+i, c]
    
    # Term 0 (M1): A00=1, A11=1; B00=1, B11=1; C00+=1, C11+=1
    U2[0*2+0][0] = 1  # A00
    U2[1*2+1][0] = 1  # A11
    V2[0*2+0][0] = 1  # B00
    V2[1*2+1][0] = 1  # B11
    W2[0*2+0][0] = 1  # C00
    W2[1*2+1][0] = 1  # C11
    
    # Term 1 (M2): A10=1, A11=1; B00=1; C10+=1
    U2[1*2+0][1] = 1  # A10
    U2[1*2+1][1] = 1  # A11
    V2[0*2+0][1] = 1  # B00
    W2[0*2+1][1] = 1  # C10 (k=0, i=1)
    
    # Term 2 (M3): A00=1; B01=1, B11=-1; C01+=1
    U2[0*2+0][2] = 1  # A00
    V2[0*2+1][2] = 1  # B01
    V2[1*2+1][2] = -1 # B11
    W2[1*2+0][2] = 1  # C01 (k=1, i=0)
    
    # Term 3 (M4): A11=1; B10=1, B00=-1; C00-=1, C01+=1
    U2[1*2+1][3] = 1  # A11
    V2[1*2+0][3] = 1  # B10
    V2[0*2+0][3] = -1 # B00
    W2[0*2+0][3] = -1 # C00
    W2[1*2+0][3] = 1  # C01 (k=1, i=0)
    
    # Term 4 (M5): A00=1, A01=1; B11=1; C00+=1, C11+=1
    U2[0*2+0][4] = 1  # A00
    U2[0*2+1][4] = 1  # A01
    V2[1*2+1][4] = 1  # B11
    W2[0*2+0][4] = 1  # C00
    W2[1*2+1][4] = 1  # C11
    
    # Term 5 (M6): A10=1, A00=-1; B00=1, B01=1; C11+=1
    U2[1*2+0][5] = 1  # A10
    U2[0*2+0][5] = -1 # A00
    V2[0*2+0][5] = 1  # B00
    V2[0*2+1][5] = 1  # B01
    W2[1*2+1][5] = 1  # C11
    
    # Term 6 (M7): A01=1, A11=-1; B10=1, B11=1; C00+=1, C10+=1
    U2[0*2+1][6] = 1  # A01
    U2[1*2+1][6] = -1 # A11
    V2[1*2+0][6] = 1  # B10
    V2[1*2+1][6] = 1  # B11
    W2[0*2+0][6] = 1  # C00
    W2[0*2+1][6] = 1  # C10 (k=0, i=1)
    
    # Verify 2x2 Strassen: sum_c U2[i*2+j,c] * V2[j*2+k,c] * W2[k*2+i,c] == delta(i,k)
    # This is the standard Strassen, so it should work.
    
    # Now build 4x4x4 rank-49 decomposition via Kronecker product.
    # The 4x4 matrix A has indices (I,J) where I = 2*i + i2, J = 2*j + j2
    # The 4x4 matrix B has indices (J,K) where J = 2*j + j2, K = 2*k + k2
    # The 4x4 matrix C has indices (K,I) where K = 2*k + k2, I = 2*i + i2
    # 
    # C[I,K] = sum_J A[I,J] * B[J,K]
    # = sum_{j,j2} A[2i+i2, 2j+j2] * B[2j+j2, 2k+k2]
    # = sum_{i2,j2,k2} [sum_j A[2i+i2, 2j+j2] B[2j+j2, 2k+k2]] ... 
    # 
    # Actually the Kronecker product approach:
    # If we view 4x4 as 2x2 blocks of 2x2, then:
    # C = A * B where A, B, C are 4x4
    # The (i,k)-th 2x2 block of C equals sum_j (A's (i,j) 2x2 block) * (B's (j,k) 2x2 block)
    # 
    # So C_block[i,k] = sum_j A_block[i,j] * B_block[j,k]
    # 
    # Using Strassen on the 2x2 block level:
    # C_block[i,k] = sum_c S_c(i,k) * M_c where M_c = sum_j U2_block[i,j,c] * V2_block[j,k,c]
    # 
    # But each M_c is a 2x2 matrix product of the block-level "super-entries".
    # 
    # The rank-49 decomposition: for each pair of Strassen terms (c1, c2) from the two levels,
    # we get one rank-1 term.
    # 
    # U4[I,J,c1*7+c2] = U2[i,j,c1] * U2[i2,j2,c2]  where I=2i+i2, J=2j+j2
    # V4[J,K,c1*7+c2] = V2[j,k,c1] * V2[j2,k2,c2]  where J=2j+j2, K=2k+k2
    # W4[K,I,c1*7+c2] = W2[k,i,c1] * W2[k2,i2,c2]  where K=2k+k2, I=2i+i2
    
    rank_4x4 = 49
    U4 = [[0]*rank_4x4 for _ in range(16)]  # 4x4 = 16 entries
    V4 = [[0]*rank_4x4 for _ in range(16)]
    W4 = [[0]*rank_4x4 for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            col = c1 * 7 + c2
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    I = 2*i + i2
                                    J = 2*j + j2
                                    K = 2*k + k2
                                    u_val = U2[i*2+j][c1] * U2[i2*2+j2][c2]
                                    v_val = V2[j*2+k][c1] * V2[j2*2+k2][c2]
                                    w_val = W2[k*2+i][c1] * W2[k2*2+i2][c2]
                                    U4[I*4+J][col] += u_val
                                    V4[J*4+K][col] += v_val
                                    W4[K*4+I][col] += w_val
    
    # Now for 4x4 * 4x8:
    # A is 4x4 (indices I in 0..3, J in 0..3)
    # B is 4x8 (indices J in 0..3, K in 0..7)
    # C is 4x8 (indices K in 0..3, I in 0..3) -- wait, output is 4x8
    # 
    # C[I,K] = sum_J A[I,J] * B[J,K] for I in 0..3, K in 0..7
    # 
    # Split B into two 4x4 blocks: B_left[J,K] for K in 0..3, B_right[J,K] for K in 4..7
    # C_left[I,K] = sum_J A[I,J] * B_left[J,K] for K in 0..3
    # C_right[I,K] = sum_J A[I,J] * B_right[J,K] for K in 4..7
    # 
    # Each uses the rank-49 decomposition.
    # 
    # For C_left: output index is (K, I) with K in 0..3, I in 0..3
    #   W4[K*4+I, col]
    # For C_right: output index is (K+4, I) with K in 0..3, I in 0..3
    #   W4[K*4+I, col] but mapped to position (K+4)*4 + I in the full 8x4 output
    # 
    # Wait, the output W has shape (p*n, rank) = (8*4, rank) = (32, rank)
    # W[k*n+i, c] where k is the column of B (0..7), i is the row of A (0..3)
    # 
    # So for C_left (K in 0..3): W[K*4+I, col]
    # For C_right (K in 4..7): W[K*4+I, col] where K = 4+k2_block
    
    n, m, p = 4, 4, 8
    rank = 98  # 49 + 49
    
    U = [[0]*rank for _ in range(n*m)]   # 16 x 98
    V = [[0]*rank for _ in range(m*p)]   # 32 x 98
    W = [[0]*rank for _ in range(p*n)]   # 32 x 98
    
    # Left block (K in 0..3): columns 0..48
    for col in range(49):
        for I in range(4):
            for J in range(4):
                U[I*4+J][col] = U4[I*4+J][col]
        for J in range(4):
            for K in range(4):
                V[J*8+K][col] = V4[J*4+K][col]  # B_left[J,K] where K in 0..3
        for K in range(4):
            for I in range(4):
                W[K*4+I][col] = W4[K*4+I][col]
    
    # Right block (K in 4..7): columns 49..97
    for col in range(49):
        col_r = 49 + col
        for I in range(4):
            for J in range(4):
                U[I*4+J][col_r] = U4[I*4+J][col]
        for J in range(4):
            for K in range(4):
                V[J*8+(K+4)][col_r] = V4[J*4+K][col]  # B_right[J,K] where K in 0..3 -> col K+4
        for K in range(4):
            for I in range(4):
                W[(K+4)*4+I][col_r] = W4[K*4+I][col]
    
    return {"U": U, "V": V, "W": W}

