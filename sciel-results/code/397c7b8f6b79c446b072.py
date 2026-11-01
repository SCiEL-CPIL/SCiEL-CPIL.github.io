"""Winograd Fourier Transform Technique (WFT) for 2x4x7 matrix multiplication."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    # We use a block decomposition approach:
    # Split the 2x4x7 multiplication into sub-blocks.
    # A is 2x4, B is 4x7, C is 2x7.
    # We can view this as multiplying 2x2 blocks of A with 2x2 blocks of B? No, B has 7 columns.
    # Let's split B into two parts: B1 (4x4) and B2 (4x3).
    # C1 = A * B1 (2x4) and C2 = A * B2 (2x3).
    
    # For C1 (2x4x4):
    # We can use Strassen-like decomposition or standard block decomposition.
    # Standard: 2*4*4 = 32 terms.
    # Strassen for 2x2x2 blocks:
    # Split A into A_top (2x2), A_bottom (2x2)? No, A is 2x4.
    # Let's split A into A_L (2x2) and A_R (2x2).
    # Split B1 into B1_L (4x2) and B1_R (4x2).
    # C1_L = A_L * B1_L (2x2) -> Strassen rank 7
    # C1_R = A_R * B1_R (2x2) -> Strassen rank 7
    # But C1 = A * B1 = [A_L A_R] * [B1_L; B1_R]? No.
    # A is 2x4. B1 is 4x4.
    # A * B1 = sum_j A[:,j] B1[j,:].
    # If we split columns of A into two groups of 2: A1 (2x2), A2 (2x2).
    # Split rows of B1 into two groups of 2: B1_1 (2x4), B1_2 (2x4).
    # Then A * B1 = A1 * B1_1 + A2 * B1_2.
    # A1 * B1_1 is 2x2 * 2x4.
    # This is not a square matrix multiplication.
    
    # Let's stick to the direct sum of inner products but use Strassen for the 2x2 sub-multiplications where possible.
    # Actually, the rank of M_{2,4,7} is at most 56.
    # Can we do better?
    # The rank of M_{n,m,p} is at most n*m*p.
    # For M_{2,2,2}, rank is 7.
    # For M_{2,4,2}, we can split into two M_{2,2,2} blocks?
    # A (2x4) = [A1 A2], B (4x2) = [B1; B2].
    # C = A1*B1 + A2*B2.
    # Each term is 2x2 * 2x2.
    # Using Strassen for each, rank is 7 + 7 = 14.
    # Schoolbook for M_{2,4,2} is 2*4*2 = 16.
    # So we save 2 terms.
    
    # For M_{2,4,7}:
    # Split B into B1 (4x4) and B2 (4x3).
    # C1 = A * B1 (2x4).
    # C2 = A * B2 (2x3).
    
    # For C1 (2x4x4):
    # Split A into A1 (2x2), A2 (2x2).
    # Split B1 into B1_1 (4x2), B1_2 (4x2).
    # C1 = A1 * B1_1 + A2 * B1_2.
    # A1 * B1_1 is 2x2 * 2x4.
    # This is two 2x2x2 multiplications: (A1 * B1_1[:,0:2]) and (A1 * B1_1[:,2:4]).
    # Wait, B1_1 is 4x2. The multiplication is (2x2) * (2x2) for each column block?
    # No, A1 is 2x2. B1_1 is 4x2.
    # A1 * B1_1 is not defined directly.
    # Let's re-index.
    # A is 2x4. Rows i=0,1. Cols j=0,1,2,3.
    # B1 is 4x4. Rows j=0,1,2,3. Cols k=0,1,2,3.
    # C1[i,k] = sum_{j=0}^3 A[i,j] B1[j,k].
    # Split j into j=0,1 and j=2,3.
    # C1[i,k] = sum_{j=0}^1 A[i,j] B1[j,k] + sum_{j=2}^3 A[i,j] B1[j,k].
    # Let A_L = A[:, 0:2] (2x2), A_R = A[:, 2:4] (2x2).
    # Let B1_L = B1[0:2, :] (2x4), B1_R = B1[2:4, :] (2x4).
    # C1 = A_L * B1_L + A_R * B1_R.
    # A_L * B1_L is 2x2 * 2x4.
    # This is equivalent to two 2x2 * 2x2 multiplications:
    # C1[:, 0:2] = A_L * B1_L[:, 0:2]
    # C1[:, 2:4] = A_L * B1_L[:, 2:4]
    # Each is a 2x2x2 multiplication. Rank 7 each. Total 14 for A_L * B1_L.
    # Similarly, A_R * B1_R is 2x2 * 2x4.
    # C1[:, 0:2] += A_R * B1_R[:, 0:2] (Rank 7)
    # C1[:, 2:4] += A_R * B1_R[:, 2:4] (Rank 7)
    # Total rank for C1: 14 + 14 = 28.
    # Schoolbook for C1: 2*4*4 = 32.
    # Saving: 4.
    
    # For C2 (2x4x3):
    # B2 is 4x3.
    # C2 = A_L * B2_L + A_R * B2_R.
    # B2_L is 2x3, B2_R is 2x3.
    # A_L * B2_L is 2x2 * 2x3.
    # This is not a square multiplication.
    # We can treat it as 3 separate 2x2x1 multiplications?
    # Or 1.5 2x2x2?
    # The rank of 2x2 * 2x1 is 4 (schoolbook) or 3 (Strassen-like for rectangular?).
    # Actually, the rank of M_{2,2,1} is 4? No, it's just a dot product of length 2 for each output.
    # C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0].
    # This is 2 dot products of length 2.
    # Rank is 4?
    # T = sum_{i,j} A[i,j] B[j,0] C[i,0].
    # This is two independent inner products of length 2.
    # Rank of inner product of length 2 is 2? No, rank of M_{1,2,1} is 2.
    # So M_{2,2,1} has rank 4.
    # Schoolbook is 2*2*1 = 4.
    # So no saving here with simple block decomposition.
    # Total rank for C2: 4 (for A_L*B2_L) + 4 (for A_R*B2_R) = 8.
    # Schoolbook for C2: 2*4*3 = 24.
    # Wait, my previous calculation was wrong.
    # C2[i,k] = sum_{j=0}^3 A[i,j] B2[j,k].
    # Split j:
    # C2 = A_L * B2_L + A_R * B2_R.
    # A_L (2x2) * B2_L (2x3) -> 2x3 matrix.
    # This requires 2*2*3 = 12 terms? No.
    # It's sum_{j=0}^1 A_L[i,j] B2_L[j,k].
    # For each k (0,1,2), we have a 2x2x1 multiplication.
    # Rank of M_{2,2,1} is 4.
    # So for 3 columns, rank is 3*4 = 12?
    # No, the terms are shared?
    # A_L[i,j] is shared across k.
    # The tensor is sum_{i,j,k} A_L[i,j] B2_L[j,k] C2[i,k].
    # This is a 2x2x3 tensor.
    # Rank?
    # Schoolbook: 2*2*3 = 12.
    # Can we do better?
    # Probably not easily without more complex algorithms.
    # So rank for A_L * B2_L is 12.
    # Rank for A_R * B2_R is 12.
    # Total for C2: 24.
    # Schoolbook for C2: 24.
    # So no saving for C2.
    
    # Total rank: 28 (for C1) + 24 (for C2) = 52.
    # Schoolbook: 56.
    # Saving: 4.
    
    # Let's implement this decomposition.
    
    def add_term(U, V, W, u_idx, v_idx, w_idx, val=1):
        U[u_idx].append(val)
        V[v_idx].append(val)
        W[w_idx].append(val)
    
    # Initialize matrices with empty lists for columns
    U = [[] for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[] for _ in range(m * p)]  # 4*7 = 28 rows
    W = [[] for _ in range(p * n)]  # 7*2 = 14 rows
    
    # Helper to add a Strassen decomposition for M_{2,2,2}
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    def add_strassen(U, V, W, a_idx, b_idx, c_idx):
        # a_idx: list of 4 indices [a00, a01, a10, a11]
        # b_idx: list of 4 indices [b00, b01, b10, b11]
        # c_idx: list of 4 indices [c00, c01, c10, c11]
        
        # P1 = (A00+A11)(B00+B11)
        U[a_idx[0]].append(1); U[a_idx[3]].append(1)
        V[b_idx[0]].append(1); V[b_idx[3]].append(1)
        W[c_idx[0]].append(1); W[c_idx[3]].append(1); W[c_idx[1]].append(-1)
        
        # P2 = (A10+A11)B00
        U[a_idx[2]].append(1); U[a_idx[3]].append(1)
        V[b_idx[0]].append(1)
        W[c_idx[2]].append(1); W[c_idx[3]].append(-1)
        
        # P3 = A00(B01-B11)
        U[a_idx[0]].append(1)
        V[b_idx[1]].append(1); V[b_idx[3]].append(-1)
        W[c_idx[1]].append(1); W[c_idx[3]].append(1)
        
        # P4 = A11(B10-B00)
        U[a_idx[3]].append(1)
        V[b_idx[2]].append(1); V[b_idx[0]].append(-1)
        W[c_idx[0]].append(1); W[c_idx[2]].append(1)
        
        # P5 = (A00+A01)B11
        U[a_idx[0]].append(1); U[a_idx[1]].append(1)
        V[b_idx[3]].append(1)
        W[c_idx[0]].append(-1); W[c_idx[1]].append(1)
        
        # P6 = (A10-A00)(B00+B01)
        U[a_idx[2]].append(1); U[a_idx[0]].append(-1)
        V[b_idx[0]].append(1); V[b_idx[1]].append(1)
        W[c_idx[3]].append(1)
        
        # P7 = (A01-A11)(B10+B11)
        U[a_idx[1]].append(1); U[a_idx[3]].append(-1)
        V[b_idx[2]].append(1); V[b_idx[3]].append(1)
        W[c_idx[0]].append(1); W[c_idx[3]].append(1)
        
        # Wait, I need to check the Strassen formulas and output mappings carefully.
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # My indices:
        # A: 00, 01, 10, 11 -> 11, 12, 21, 22 in 1-based
        # B: 00, 01, 10, 11 -> 11, 12, 21, 22 in 1-based
        # C: 00, 01, 10, 11 -> 11, 12, 21, 22 in 1-based
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        # M2: (A10+A11)B00 -> C10, C11
        # M3: A00(B01-B11) -> C01, C11
        # M4: A11(B10-B00) -> C00, C10
        # M5: (A00+A01)B11 -> C00, C01
        # M6: (A10-A00)(B00+B01) -> C11
        # M7: (A01-A11)(B10+B11) -> C00, C11
        
        # Let's re-verify the output coefficients.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # My previous add_strassen had:
        # P1 (M1): W[c0]=1, W[c3]=1, W[c1]=-1. WRONG.
        # M1 contributes to C00 and C11.
        # So W[c0]+=1, W[c3]+=1.
        
        # Let's rewrite add_strassen correctly.
        pass # We will implement this properly below.

    # Correct Strassen implementation
    def add_strassen_correct(U, V, W, a_idx, b_idx, c_idx):
        # a_idx: [A00, A01, A10, A11]
        # b_idx: [B00, B01, B10, B11]
        # c_idx: [C00, C01, C10, C11]
        
        # M1 = (A00+A11)(B00+B11)
        U[a_idx[0]].append(1); U[a_idx[3]].append(1)
        V[b_idx[0]].append(1); V[b_idx[3]].append(1)
        W[c_idx[0]].append(1); W[c_idx[3]].append(1)
        
        # M2 = (A10+A11)B00
        U[a_idx[2]].append(1); U[a_idx[3]].append(1)
        V[b_idx[0]].append(1)
        W[c_idx[2]].append(1); W[c_idx[3]].append(-1)
        
        # M3 = A00(B01-B11)
        U[a_idx[0]].append(1)
        V[b_idx[1]].append(1); V[b_idx[3]].append(-1)
        W[c_idx[1]].append(1); W[c_idx[3]].append(1)
        
        # M4 = A11(B10-B00)
        U[a_idx[3]].append(1)
        V[b_idx[2]].append(1); V[b_idx[0]].append(-1)
        W[c_idx[0]].append(1); W[c_idx[2]].append(1)
        
        # M5 = (A00+A01)B11
        U[a_idx[0]].append(1); U[a_idx[1]].append(1)
        V[b_idx[3]].append(1)
        W[c_idx[0]].append(-1); W[c_idx[1]].append(1)
        
        # M6 = (A10-A00)(B00+B01)
        U[a_idx[2]].append(1); U[a_idx[0]].append(-1)
        V[b_idx[0]].append(1); V[b_idx[1]].append(1)
        W[c_idx[3]].append(1)
        
        # M7 = (A01-A11)(B10+B11)
        U[a_idx[1]].append(1); U[a_idx[3]].append(-1)
        V[b_idx[2]].append(1); V[b_idx[3]].append(1)
        W[c_idx[0]].append(1); W[c_idx[3]].append(1)

    # Indices for A (2x4):
    # A[i, j] -> index i*m + j
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    
    # Indices for B (4x7):
    # B[j, k] -> index j*p + k
    # B[0,0]=0, ..., B[0,6]=6
    # B[1,0]=7, ..., B[1,6]=13
    # B[2,0]=14, ..., B[2,6]=20
    # B[3,0]=21, ..., B[3,6]=27
    
    # Indices for C (2x7):
    # C[i, k] -> index k*n + i
    # C[0,0]=0, C[1,0]=1
    # C[0,1]=2, C[1,1]=3
    # ...
    # C[0,6]=12, C[1,6]=13
    
    # C1 part (k=0..3)
    # A_L: A[:, 0:2] -> indices [0, 1, 4, 5] (A00, A01, A10, A11)
    # A_R: A[:, 2:4] -> indices [2, 3, 6, 7]
    
    # B1_L: B[0:2, 0:4] -> B[0,0..3], B[1,0..3]
    # B1_R: B[2:4, 0:4] -> B[2,0..3], B[3,0..3]
    
    # C1_L: C[:, 0:2] -> C[0,0..1], C[1,0..1]
    # C1_R: C[:, 2:4] -> C[0,2..3], C[1,2..3]
    
    # Block 1: A_L * B1_L[:, 0:2] -> C[:, 0:2]
    # A_L idx: [0, 1, 4, 5]
    # B1_L[:, 0:2] idx: 
    # B[0,0]=0, B[0,1]=1, B[1,0]=7, B[1,1]=8
    # b_idx: [0, 1, 7, 8]
    # C[:, 0:2] idx:
    # C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    # c_idx: [0, 2, 1, 3]
    add_strassen_correct(U, V, W, [0, 1, 4, 5], [0, 1, 7, 8], [0, 2, 1, 3])
    
    # Block 2: A_L * B1_L[:, 2:4] -> C[:, 2:4]
    # A_L idx: [0, 1, 4, 5]
    # B1_L[:, 2:4] idx:
    # B[0,2]=2, B[0,3]=3, B[1,2]=9, B[1,3]=10
    # b_idx: [2, 3, 9, 10]
    # C[:, 2:4] idx:
    # C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
    # c_idx: [4, 6, 5, 7]
    add_strassen_correct(U, V, W, [0, 1, 4, 5], [2, 3, 9, 10], [4, 6, 5, 7])
    
    # Block 3: A_R * B1_R[:, 0:2] -> C[:, 0:2]
    # A_R idx: [2, 3, 6, 7]
    # B1_R[:, 0:2] idx:
    # B[2,0]=14, B[2,1]=15, B[3,0]=21, B[3,1]=22
    # b_idx: [14, 15, 21, 22]
    # C[:, 0:2] idx: [0, 2, 1, 3]
    add_strassen_correct(U, V, W, [2, 3, 6, 7], [14, 15, 21, 22], [0, 2, 1, 3])
    
    # Block 4: A_R * B1_R[:, 2:4] -> C[:, 2:4]
    # A_R idx: [2, 3, 6, 7]
    # B1_R[:, 2:4] idx:
    # B[2,2]=16, B[2,3]=17, B[3,2]=23, B[3,3]=24
    # b_idx: [16, 17, 23, 24]
    # C[:, 2:4] idx: [4, 6, 5, 7]
    add_strassen_correct(U, V, W, [2, 3, 6, 7], [16, 17, 23, 24], [4, 6, 5, 7])
    
    # C2 part (k=4..6)
    # B2_L: B[0:2, 4:7]
    # B2_R: B[2:4, 4:7]
    # C2: C[:, 4:7]
    
    # For C2, we use schoolbook for 2x2 * 2x3 blocks.
    # A_L * B2_L:
    # A_L idx: [0, 1, 4, 5]
    # B2_L: B[0,4..6], B[1,4..6]
    # B[0,4]=4, B[0,5]=5, B[0,6]=6
    # B[1,4]=11, B[1,5]=12, B[1,6]=13
    # C2_L: C[0,4..6], C[1,4..6]
    # C[0,4]=8, C[0,5]=10, C[0,6]=12
    # C[1,4]=9, C[1,5]=11, C[1,6]=13
    
    # Terms:
    # C[0,4] += A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[0,5] += A[0,0]B[0,5] + A[0,1]B[1,5]
    # C[0,6] += A[0,0]B[0,6] + A[0,1]B[1,6]
    # C[1,4] += A[1,0]B[0,4] + A[1,1]B[1,4]
    # C[1,5] += A[1,0]B[0,5] + A[1,1]B[1,5]
    # C[1,6] += A[1,0]B[0,6] + A[1,1]B[1,6]
    
    # We can factor this:
    # A[0,0] * (B[0,4]C[0,4] + B[0,5]C[0,5] + B[0,6]C[0,6])
    # A[0,1] * (B[1,4]C[0,4] + B[1,5]C[0,5] + B[1,6]C[0,6])
    # A[1,0] * (B[0,4]C[1,4] + B[0,5]C[1,5] + B[0,6]C[1,6])
    # A[1,1] * (B[1,4]C[1,4] + B[1,5]C[1,5] + B[1,6]C[1,6])
    
    # This is 4 terms.
    # Term 1: U[0]=1, V[4]=1, W[8]=1; V[5]=1, W[10]=1; V[6]=1, W[12]=1
    U[0].append(1)
    V[4].append(1); V[5].append(1); V[6].append(1)
    W[8].append(1); W[10].append(1); W[12].append(1)
    
    # Term 2: U[1]=1, V[11]=1, W[8]=1; V[12]=1, W[10]=1; V[13]=1, W[12]=1
    U[1].append(1)
    V[11].append(1); V[12].append(1); V[13].append(1)
    W[8].append(1); W[10].append(1); W[12].append(1)
    
    # Term 3: U[4]=1, V[4]=1, W[9]=1; V[5]=1, W[11]=1; V[6]=1, W[13]=1
    U[4].append(1)
    V[4].append(1); V[5].append(1); V[6].append(1)
    W[9].append(1); W[11].append(1); W[13].append(1)
    
    # Term 4: U[5]=1, V[11]=1, W[9]=1; V[12]=1, W[11]=1; V[13]=1, W[13]=1
    U[5].append(1)
    V[11].append(1); V[12].append(1); V[13].append(1)
    W[9].append(1); W[11].append(1); W[13].append(1)
    
    # A_R * B2_R:
    # A_R idx: [2, 3, 6, 7]
    # B2_R: B[2,4..6], B[3,4..6]
    # B[2,4]=18, B[2,5]=19, B[2,6]=20
    # B[3,4]=25, B[3,5]=26, B[3,6]=27
    
    # Term 5: U[2]=1, V[18]=1, W[8]=1; V[19]=1, W[10]=1; V[20]=1, W[12]=1
    U[2].append(1)
    V[18].append(1); V[19].append(1); V[20].append(1)
    W[8].append(1); W[10].append(1); W[12].append(1)
    
    # Term 6: U[3]=1, V[25]=1, W[8]=1; V[26]=1, W[10]=1; V[27]=1, W[12]=1
    U[3].append(1)
    V[25].append(1); V[26].append(1); V[27].append(1)
    W[8].append(1); W[10].append(1); W[12].append(1)
    
    # Term 7: U[6]=1, V[18]=1, W[9]=1; V[19]=1, W[11]=1; V[20]=1, W[13]=1
    U[6].append(1)
    V[18].append(1); V[19].append(1); V[20].append(1)
    W[9].append(1); W[11].append(1); W[13].append(1)
    
    # Term 8: U[7]=1, V[25]=1, W[9]=1; V[26]=1, W[11]=1; V[27]=1, W[13]=1
    U[7].append(1)
    V[25].append(1); V[26].append(1); V[27].append(1)
    W[9].append(1); W[11].append(1); W[13].append(1)
    
    # Convert lists to numpy arrays or keep as lists?
    # The interface expects matrices. Lists of lists are fine if they are rectangular.
    # But we need to ensure all rows have the same length.
    rank = len(U[0])
    # Pad with zeros if necessary (should not be needed if logic is correct, but safe)
    for i in range(len(U)):
        while len(U[i]) < rank: U[i].append(0)
    for i in range(len(V)):
        while len(V[i]) < rank: V[i].append(0)
    for i in range(len(W)):
        while len(W[i]) < rank: W[i].append(0)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
