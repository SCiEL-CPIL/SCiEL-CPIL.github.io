"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 4x4x4, applied to two 4x4 blocks of B for 4x4x8.
    # This gives rank 49 + 49 = 98.
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 base case
    # For 2x2 matrices: C = AB
    # Strassen uses 7 multiplications:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to build the 4x4x4 Strassen decomposition.
    # The 4x4x4 case can be built by applying Strassen at the 2x2 block level.
    # A is 4x4, B is 4x4, C is 4x4.
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = [[C11, C12], [C21, C22]]
    
    # Each block multiplication is 2x2x2, which itself uses Strassen (rank 7).
    # So each of the 7 Strassen multiplications for the 4x4x4 case requires 7 scalar multiplications.
    # Total rank = 7 * 7 = 49.
    
    # Let's construct the U, V, W matrices for the 4x4x4 case.
    # U has shape (n*m, rank) = (16, 49)
    # V has shape (m*p, rank) = (16, 49)  
    # W has shape (p*n, rank) = (16, 49)
    
    # For the 4x4x4 case:
    # n_block = m_block = p_block = 2 (block size)
    # N = 2 (number of blocks in each dimension)
    
    # Strassen at block level (7 terms):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each block multiplication is 2x2x2 Strassen.
    # Let's define the 2x2x2 Strassen U,V,W first.
    
    # 2x2x2 Strassen:
    # U is (n*m, 7) = (4, 7)
    # V is (m*p, 7) = (4, 7)
    # W is (p*n, 7) = (4, 7)
    
    # M1 = (A11+A22)(B11+B22)
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3
    
    # For term t, U[i,j,t] contributes to C[i,j] via W.
    # Actually, the decomposition is: C[i,k] = sum_t U[i,j,t] * A[i,j] * V[j,k,t] * B[j,k] * W[k,i,t]
    # Wait, the convention is: W[k*n+i, r]
    
    # Let me think about this more carefully.
    # The matrix multiplication tensor T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # where i is row of A and row of C, j is col of A and row of B, k is col of B and col of C.
    # Actually the standard convention: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = delta[i,k] ... no.
    # The tensor for matrix mult: T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: C = A x B, where (A x B)[i,k] = sum_j A[i,j] B[j,k]
    # The tensor decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Wait, that doesn't look right either. Let me re-read the task.
    
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by [k*n+i, r] where k is the column index of B (and C), i is the row index of A (and C).
    # U is [n*m, rank], so U[i*m+j, r]
    # V is [m*p, rank], so V[j*p+k, r]
    # W is [p*n, rank], so W[k*n+i, r]
    
    # The decomposition is: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # because C[i,k] = sum_j A[i,j] B[j,k], so the tensor T[i,j,k] = 1 for all i,j,k.
    
    # OK so T[i,j,k] = 1 for all i,j,k.
    # And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # Let me construct the 2x2x2 Strassen decomposition.
    # n=m=p=2
    # U is (4, 7), V is (4, 7), W is (4, 7)
    
    # Strassen terms:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In terms of indices (0-based):
    # A[0,0]=A11, A[0,1]=A12, A[1,0]=A21, A[1,1]=A22
    # B[0,0]=B11, B[0,1]=B12, B[1,0]=B21, B[1,1]=B22
    # C[0,0]=C11, C[0,1]=C12, C[1,0]=C21, C[1,1]=C22
    
    # For term M1 = (A11+A22)(B11+B22):
    # U contributions: U[0*2+0, 0] = 1 (A11), U[1*2+1, 0] = 1 (A22)
    # V contributions: V[0*2+0, 0] = 1 (B11), V[0*2+1, 0] = 1 (B12), V[1*2+0, 0] = 1 (B21), V[1*2+1, 0] = 1 (B22)
    # Wait, no. V[j*p+k, r] where j is col of A / row of B, k is col of B.
    # For B11+B22: B11 is B[0,0], B22 is B[1,1]
    # So V[0*2+0, 0] = 1 (B[0,0]=B11), V[1*2+1, 0] = 1 (B[1,1]=B22)
    
    # Hmm, let me be more careful.
    # M1 = (A11+A22)(B11+B22)
    # A11 = A[0,0], A22 = A[1,1]
    # B11 = B[0,0], B22 = B[1,1]
    # So U[0*2+0, 0] = 1, U[1*2+1, 0] = 1
    # V[0*2+0, 0] = 1, V[1*2+1, 0] = 1
    # W: C11 gets +M1, so W[0*2+0, 0] = 1 (C[0,0])
    # Also C22 gets +M1, so W[1*2+1, 0] = 1 (C[1,1])
    
    # Let me just code this up directly.
    
    # 2x2x2 Strassen U,V,W (each is 4x7)
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U2[0][0] = 1  # A[0,0]
    U2[3][0] = 1  # A[1,1]
    V2[0][0] = 1  # B[0,0]
    V2[3][0] = 1  # B[1,1]
    W2[0][0] = 1  # C[0,0] += M1
    W2[3][0] = 1  # C[1,1] += M1
    
    # M2 = (A21+A22)B11
    U2[2][1] = 1  # A[1,0]
    U2[3][1] = 1  # A[1,1]
    V2[0][1] = 1  # B[0,0]
    W2[2][1] = 1  # C[1,0] += M2
    W2[3][1] = -1 # C[1,1] -= M2
    
    # M3 = A11(B12-B22)
    U2[0][2] = 1  # A[0,0]
    V2[1][2] = 1  # B[0,1]
    V2[3][2] = -1 # B[1,1]
    W2[1][2] = 1  # C[0,1] += M3
    W2[3][2] = 1  # C[1,1] += M3
    
    # M4 = A22(B21-B11)
    U2[3][3] = 1  # A[1,1]
    V2[2][3] = 1  # B[1,0]
    V2[0][3] = -1 # B[0,0]
    W2[0][3] = 1  # C[0,0] += M4
    W2[2][3] = 1  # C[1,0] += M4
    
    # M5 = (A11+A12)B22
    U2[0][4] = 1  # A[0,0]
    U2[1][4] = 1  # A[0,1]
    V2[3][4] = 1  # B[1,1]
    W2[0][4] = -1 # C[0,0] -= M5
    W2[1][4] = 1  # C[0,1] += M5
    
    # M6 = (A21-A11)(B11+B12)
    U2[2][5] = 1  # A[1,0]
    U2[0][5] = -1 # A[0,0]
    V2[0][5] = 1  # B[0,0]
    V2[1][5] = 1  # B[0,1]
    W2[3][5] = 1  # C[1,1] += M6
    
    # M7 = (A12-A22)(B21+B22)
    U2[1][6] = 1  # A[0,1]
    U2[3][6] = -1 # A[1,1]
    V2[2][6] = 1  # B[1,0]
    V2[3][6] = 1  # B[1,1]
    W2[0][6] = 1  # C[0,0] += M7
    
    # Now build the 4x4x4 Strassen decomposition.
    # Block size = 2, number of blocks per dimension = 2.
    # A is 4x4, split into 2x2 blocks:
    # A_block[i,j] is a 2x2 matrix, i,j in {0,1}
    # Similarly for B and C.
    
    # Strassen at block level (7 terms), each involving 2x2 block multiplications.
    # Each 2x2 block multiplication uses the 2x2x2 Strassen decomposition (rank 7).
    # Total rank = 7 * 7 = 49.
    
    # Let's index the 49 terms as (block_term, sub_term) where block_term in 0..6, sub_term in 0..6.
    # Column index = block_term * 7 + sub_term
    
    # For the 4x4x4 case:
    # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Block-level Strassen:
    # M1_block = (A_block[0,0] + A_block[1,1]) * (B_block[0,0] + B_block[1,1])
    # M2_block = (A_block[1,0] + A_block[1,1]) * B_block[0,0]
    # M3_block = A_block[0,0] * (B_block[0,1] - B_block[1,1])
    # M4_block = A_block[1,1] * (B_block[1,0] - B_block[0,0])
    # M5_block = (A_block[0,0] + A_block[0,1]) * B_block[1,1]
    # M6_block = (A_block[1,0] - A_block[0,0]) * (B_block[0,0] + B_block[0,1])
    # M7_block = (A_block[0,1] - A_block[1,1]) * (B_block[1,0] + B_block[1,1])
    
    # C_block[0,0] = M1_block + M4_block - M5_block + M7_block
    # C_block[0,1] = M3_block + M5_block
    # C_block[1,0] = M2_block + M4_block
    # C_block[1,1] = M1_block - M2_block + M3_block + M6_block
    
    # For each block-level term, we need to add contributions to U4, V4, W4.
    # The block-level term involves products of block-level linear combinations.
    
    # Let's define the block-level coefficients:
    # For A: coeff_A[term][block_i][block_j]
    # For B: coeff_B[term][block_i][block_j]
    # For C: coeff_C[term][block_i][block_j]
    
    # A blocks: A_block[bi, bj] where bi is block row, bj is block col
    # Each block is 2x2, so within the 4x4 matrix, block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    
    # Let's define the block-level Strassen coefficients:
    # term 0 (M1): A blocks (0,0) and (1,1); B blocks (0,0) and (1,1)
    # term 1 (M2): A blocks (1,0) and (1,1); B block (0,0)
    # term 2 (M3): A block (0,0); B blocks (0,1) and (1,1) with signs
    # etc.
    
    # Let me just enumerate:
    block_A = [
        {0: 1, 3: 1},           # M1: A[0,0] + A[1,1]
        {2: 1, 3: 1},           # M2: A[1,0] + A[1,1]
        {0: 1},                 # M3: A[0,0]
        {3: 1},                 # M4: A[1,1]
        {0: 1, 1: 1},           # M5: A[0,0] + A[0,1]
        {2: 1, 0: -1},          # M6: A[1,0] - A[0,0]
        {1: 1, 3: -1},          # M7: A[0,1] - A[1,1]
    ]
    
    block_B = [
        {0: 1, 3: 1},           # M1: B[0,0] + B[1,1]
        {0: 1},                 # M2: B[0,0]
        {1: 1, 3: -1},          # M3: B[0,1] - B[1,1]
        {2: 1, 0: -1},          # M4: B[1,0] - B[0,0]
        {3: 1},                 # M5: B[1,1]
        {0: 1, 1: 1},           # M6: B[0,0] + B[0,1]
        {2: 1, 3: 1},           # M7: B[1,0] + B[1,1]
    ]
    
    block_C = [
        {0: 1, 3: 1},           # M1: C[0,0] + C[1,1]
        {2: 1, 3: -1},          # M2: C[1,0] - C[1,1]
        {1: 1, 3: 1},           # M3: C[0,1] + C[1,1]
        {0: 1, 2: 1},           # M4: C[0,0] + C[1,0]
        {0: -1, 1: 1},          # M5: -C[0,0] + C[0,1]
        {3: 1},                 # M6: C[1,1]
        {0: 1},                 # M7: C[0,0]
    ]
    
    # Now, for each block-level term bt (0..6) and each sub-term st (0..6) from the 2x2x2 Strassen:
    # Column index = bt * 7 + st
    
    for bt in range(7):
        for st in range(7):
            col = bt * 7 + st
            
            # For U4: For each A block involved in block_A[bt], for each 2x2 element in that block,
            # use the corresponding U2 entry.
            for block_idx, coeff in block_A[bt].items():
                bi = block_idx // 2  # block row
                bj = block_idx % 2    # block col
                # The 2x2 block starts at row 2*bi, col 2*bj
                for ii in range(2):
                    for jj in range(2):
                        local_i = ii  # local row in 2x2 block
                        local_j = jj  # local col in 2x2 block
                        u2_val = U2[local_i * 2 + local_j][st]
                        if u2_val != 0:
                            global_row = 2 * bi * 2 + local_i * 2 + local_j  # i*4 + j in 4x4
                            # Wait, U4[i*4+j, col]
                            # i = 2*bi + local_i, j = 2*bj + local_j
                            global_i = 2 * bi + local_i
                            global_j = 2 * bj + local_j
                            U4[global_i * 4 + global_j][col] = coeff * u2_val
            
            # For V4: Similarly for B blocks
            for block_idx, coeff in block_B[bt].items():
                bi = block_idx // 2
                bj = block_idx % 2
                for ii in range(2):
                    for jj in range(2):
                        local_i = ii  # local row in 2x2 block (this is j in B, which is col of A / row of B)
                        local_j = jj  # local col in 2x2 block (this is k in B, which is col of B)
                        v2_val = V2[local_i * 2 + local_j][st]
                        if v2_val != 0:
                            global_i = 2 * bi + local_i  # j in 4x4
                            global_j = 2 * bj + local_j  # k in 4x4
                            V4[global_i * 4 + global_j][col] = coeff * v2_val
            
            # For W4: Similarly for C blocks
            for block_idx, coeff in block_C[bt].items():
                bi = block_idx // 2
                bj = block_idx % 2
                for ii in range(2):
                    for jj in range(2):
                        local_i = ii  # local row in 2x2 block (this is k in C, which is col of C)
                        local_j = jj  # local col in 2x2 block (this is i in C, which is row of C)
                        w2_val = W2[local_i * 2 + local_j][st]
                        if w2_val != 0:
                            global_i = 2 * bi + local_i  # k in 4x4
                            global_j = 2 * bj + local_j  # i in 4x4
                            W4[global_i * 4 + global_j][col] = coeff * w2_val
    
    # Now for the 4x4x8 case:
    # A is 4x4, B is 4x8, C is 4x8.
    # Split B into B1 (4x4) and B2 (4x4).
    # C = [C1, C2] where C1 = A*B1, C2 = A*B2.
    # Use the 4x4x4 decomposition for each.
    # Total rank = 49 + 49 = 98.
    
    n_total, m_total, p_total = 4, 4, 8
    total_rank = 98
    
    U = [[0] * total_rank for _ in range(n_total * m_total)]  # 16 x 98
    V = [[0] * total_rank for _ in range(m_total * p_total)]  # 32 x 98
    W = [[0] * total_rank for _ in range(p_total * n_total)]  # 32 x 98
    
    # First 49 columns: for B1 (cols 0-3 of B)
    for r in range(49):
        for i in range(16):
            U[i][r] = U4[i][r]
        for j in range(32):
            if j < 16:  # B1 part
                V[j][r] = V4[j][r]
            else:
                V[j][r] = 0
        for k in range(32):
            if k < 16:  # C1 part
                W[k][r] = W4[k][r]
            else:
                W[k][r] = 0
    
    # Next 49 columns: for B2 (cols 4-7 of B)
    for r in range(49):
        for i in range(16):
            U[i][49 + r] = U4[i][r]
        for j in range(32):
            if j >= 16:  # B2 part
                V[j][49 + r] = V4[j - 16][r]
            else:
                V[j][49 + r] = 0
        for k in range(32):
            if k >= 16:  # C2 part
                W[k][49 + r] = W4[k - 16][r]
            else:
                W[k][49 + r] = 0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

