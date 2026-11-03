import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    # We decompose B (m x p) into B1 (m x 4), B2 (m x 4), B3 (m x 1)
    # C = A * B1 + A * B2 + A * B3
    # Rank(A*B1) = 49 (using Strassen for 4x4x4)
    # Rank(A*B2) = 49
    # Rank(A*B3) = 16 (since B3 is 4x1, it's just 4 matrix-vector products, each rank 4? No, 4x4x1 is rank 16? Let's check: A is 4x4, v is 4x1. C = A v. This is a bilinear map. Rank is 4*4 = 16? Yes, because each entry C_ij = sum_k A_ik v_k. The tensor is sum_k e_i e_k^T e_k e_j? No.
    # Tensor T_{i,k,j} = delta_{i,k}? No.
    # C_{i,j} = sum_k A_{i,k} B_{k,j}.
    # If B is 4x1, j is fixed (say j=0).
    # C_{i,0} = sum_k A_{i,k} B_{k,0}.
    # This is 4 independent dot products of length 4.
    # Each dot product has rank 4. Total rank 16.
    # So total rank = 49 + 49 + 16 = 114.
    
    rank = 114
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen decomposition for 2x2x2
    # Factors are (4, 7) for U, (4, 7) for V, (4, 7) for W
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # The 7 rank-1 terms for Strassen:
    # M1 = (A11)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B21)
    # M4 = (A22)(B22-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B12+B11)
    # M7 = (A12-A22)(B21+B22)
    
    # We need to construct the 49 terms for 4x4x4.
    # This is effectively 7x7 combinations of the 2x2 factors.
    # Let's define the 2x2 factors explicitly.
    
    # U_2x2: 4x7
    U_2 = np.zeros((4, 7), dtype=float)
    V_2 = np.zeros((4, 7), dtype=float)
    W_2 = np.zeros((4, 7), dtype=float)
    
    # Term 1: (A11)(B11+B22)
    # A11 is at (0,0) -> idx 0
    # B11 is at (0,0) -> idx 0, B22 is at (1,1) -> idx 3
    # C contribution: C11 += M1, C22 += M1?
    # C11 = M1 - M4 - M6 + M7
    # C12 = M5 + M6
    # C21 = M2 + M3
    # C22 = M1 - M3 + M4 + M5
    
    # Let's just use the standard Strassen matrix form.
    # U_2[i, r], V_2[j, r], W_2[k, r]
    
    # r=0: M1
    # A: A11 -> U[0,0]=1
    # B: B11, B22 -> V[0,0]=1, V[3,0]=1
    # C: C11, C22 -> W[0,0]=1, W[3,0]=1
    U_2[0, 0] = 1
    V_2[0, 0] = 1
    V_2[3, 0] = 1
    W_2[0, 0] = 1
    W_2[3, 0] = 1
    
    # r=1: M2
    # A: A21, A22 -> U[2,1]=1, U[3,1]=1
    # B: B11 -> V[0,1]=1
    # C: C21, C22 -> W[2,1]=1, W[3,1]=1
    U_2[2, 1] = 1
    U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1
    W_2[3, 1] = 1
    
    # r=2: M3
    # A: A11 -> U[0,2]=1
    # B: B21 -> V[2,2]=1
    # C: C21, C22 -> W[2,2]=-1, W[3,2]=-1
    U_2[0, 2] = 1
    V_2[2, 2] = 1
    W_2[2, 2] = -1
    W_2[3, 2] = -1
    
    # r=3: M4
    # A: A22 -> U[3,3]=1
    # B: B22, B11 -> V[3,3]=1, V[0,3]=-1
    # C: C11, C22 -> W[0,3]=-1, W[3,3]=1
    U_2[3, 3] = 1
    V_2[3, 3] = 1
    V_2[0, 3] = -1
    W_2[0, 3] = -1
    W_2[3, 3] = 1
    
    # r=4: M5
    # A: A11, A12 -> U[0,4]=1, U[1,4]=1
    # B: B22 -> V[3,4]=1
    # C: C12, C22 -> W[1,4]=1, W[3,4]=1
    U_2[0, 4] = 1
    U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[1, 4] = 1
    W_2[3, 4] = 1
    
    # r=5: M6
    # A: A21, A11 -> U[2,5]=1, U[0,5]=-1
    # B: B12, B11 -> V[1,5]=1, V[0,5]=1
    # C: C11, C12 -> W[0,5]=-1, W[1,5]=1
    U_2[2, 5] = 1
    U_2[0, 5] = -1
    V_2[1, 5] = 1
    V_2[0, 5] = 1
    W_2[0, 5] = -1
    W_2[1, 5] = 1
    
    # r=6: M7
    # A: A12, A22 -> U[1,6]=1, U[3,6]=-1
    # B: B21, B22 -> V[2,6]=1, V[3,6]=1
    # C: C11, C22 -> W[0,6]=1, W[3,6]=1
    U_2[1, 6] = 1
    U_2[3, 6] = -1
    V_2[2, 6] = 1
    V_2[3, 6] = 1
    W_2[0, 6] = 1
    W_2[3, 6] = 1
    
    # Now construct 4x4x4 factors from 2x2x2 factors.
    # The 4x4 matrix is a 2x2 block matrix of 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # The multiplication C = AB can be expressed using the 7 Strassen multiplications
    # but applied to the blocks.
    # However, the blocks themselves are 2x2 matrices, so their multiplication
    # is not just a scalar multiplication.
    # Wait, the standard way to get rank 49 for 4x4x4 is to use the 7x7 outer product
    # of the Strassen factors?
    # Yes. The tensor for 4x4x4 is the tensor product of the 2x2x4 tensor with itself?
    # No.
    # Let's use the explicit construction.
    # For each pair of Strassen terms (r1, r2), we get a term in the 4x4x4 decomposition.
    # There are 7*7 = 49 terms.
    
    # U_4 is 16x49
    # V_4 is 16x49
    # W_4 is 16x49
    
    # The indices for 4x4 matrices are:
    # A: (i, j) where i, j in 0..3.
    # Block structure: i = 2*I + i2, j = 2*J + j2.
    # So A[i, j] is in block (I, J) at position (i2, j2).
    
    # For a term (r1, r2):
    # U_4[2*I + i2, 2*J + j2, r1*7 + r2] = U_2[I*2+J, r1] * U_2[i2*2+j2, r2]
    # Wait, U_2 indices are (block_row, block_col, strassen_idx).
    # U_2 shape is (4, 7). Index is (block_idx, strassen_idx).
    # block_idx = I*2 + J.
    # So U_2[I*2+J, r1] is the coefficient for block (I, J) in strassen term r1.
    # Similarly for the sub-block.
    
    U_4 = np.zeros((16, 49), dtype=float)
    V_4 = np.zeros((16, 49), dtype=float)
    W_4 = np.zeros((16, 49), dtype=float)
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            # Iterate over blocks
            for I in range(2):
                for J in range(2):
                    # Block (I, J)
                    u_block = U_2[I * 2 + J, r1]
                    if u_block == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u_sub = U_2[i2 * 2 + j2, r2]
                            if u_sub == 0: continue
                            idx_u = (I * 2 + i2) * 4 + (J * 2 + j2) # row*4+col
                            U_4[idx_u, col] = u_block * u_sub
                            
            for J in range(2):
                for K in range(2):
                    v_block = V_2[J * 2 + K, r1]
                    if v_block == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v_sub = V_2[j2 * 2 + k2, r2]
                            if v_sub == 0: continue
                            idx_v = (J * 2 + j2) * 4 + (K * 2 + k2)
                            V_4[idx_v, col] = v_block * v_sub
                            
            for K in range(2):
                for I in range(2):
                    w_block = W_2[K * 2 + I, r1]
                    if w_block == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w_sub = W_2[k2 * 2 + i2, r2]
                            if w_sub == 0: continue
                            idx_w = (K * 2 + k2) * 4 + (I * 2 + i2)
                            W_4[idx_w, col] = w_block * w_sub
    
    # Now we have U_4, V_4, W_4 for 4x4x4.
    # We need to use this for A * B1 and A * B2.
    # B1 is columns 0-3 of B.
    # B2 is columns 4-7 of B.
    # B3 is column 8 of B.
    
    # For A * B1:
    # U is same as U_4.
    # V is V_4, but mapped to the first 4 columns of B.
    # B1 has shape 4x4. Its indices are (j, k) where j in 0..3, k in 0..3.
    # V_4 indices are (j, k) where j in 0..3, k in 0..3.
    # So V_B1 = V_4.
    # W is W_4, mapped to the first 4 columns of C.
    # C1 has shape 4x4. Its indices are (k, i) where k in 0..3, i in 0..3.
    # W_4 indices are (k, i) where k in 0..3, i in 0..3.
    # So W_C1 = W_4.
    
    # For A * B2:
    # U is same as U_4.
    # V is V_4, mapped to columns 4-7 of B.
    # B2 has shape 4x4. Its indices are (j, k) where j in 0..3, k in 0..3.
    # But in the full B, these are (j, 4+k).
    # So V_B2[j, 4+k] = V_4[j, k].
    # W is W_4, mapped to columns 4-7 of C.
    # C2 has shape 4x4. Its indices are (k, i) where k in 0..3, i in 0..3.
    # But in the full C, these are (4+k, i).
    # So W_C2[4+k, i] = W_4[k, i].
    
    # For A * B3:
    # B3 is 4x1.
    # U is 4x4 matrix.
    # V is 4x1 vector.
    # W is 4x1 vector.
    # Rank 16.
    # U_B3[i, r] = A[i, k]
    # V_B3[k, r] = B[k, 8]
    # W_B3[8, i] = C[i, 8]
    # So for each i, k:
    # U_B3[i, i*4+k] = 1
    # V_B3[k, i*4+k] = 1
    # W_B3[8, i*4+k] = 1
    
    # Let's populate the main U, V, W arrays.
    
    # Part 1: A * B1 (cols 0-3 of C)
    # Rank 49
    U[:, 0:49] = U_4
    # V_4 is 16x49. Indices (j, k).
    # B1 is cols 0-3. So V[j, k] = V_4[j, k].
    V[:, 0:49] = V_4
    # W_4 is 16x49. Indices (k, i).
    # C1 is cols 0-3. So W[k, i] = W_4[k, i].
    # But W is p*n = 9*4 = 36.
    # W_4 is 16x49.
    # W[0:16, 0:49] = W_4
    W[0:16, 0:49] = W_4
    
    # Part 2: A * B2 (cols 4-7 of C)
    # Rank 49
    U[:, 49:98] = U_4
    # V_B2[j, 4+k] = V_4[j, k]
    # So V[:, 49:98] should be V_4 but shifted in columns.
    # V is 36x114.
    # V[j, 49+r] = V_4[j, r] for j in 0..3, r in 0..48?
    # No, V_4 is 16x49.
    # V_4 index is (j*4+k, r).
    # We want V[j*4 + (4+k), 49+r] = V_4[j*4+k, r].
    # Let's construct V_B2.
    V_B2 = np.zeros((16, 49), dtype=float)
    for j in range(4):
        for k in range(4):
            V_B2[j*4 + (4+k), :] = V_4[j*4+k, :]
    V[:, 49:98] = V_B2
    
    # W_C2[4+k, i] = W_4[k, i]
    # W is 36x114.
    # W_4 index is (k*4+i, r).
    # We want W[(4+k)*4+i, 49+r] = W_4[k*4+i, r].
    W_C2 = np.zeros((16, 49), dtype=float)
    for k in range(4):
        for i in range(4):
            W_C2[(4+k)*4+i, :] = W_4[k*4+i, :]
    W[:, 49:98] = W_C2
    
    # Part 3: A * B3 (col 8 of C)
    # Rank 16
    # U_B3[i, r] = 1 if r = i*4+k
    # V_B3[k, r] = 1 if r = i*4+k
    # W_B3[8, r] = 1 if r = i*4+k
    
    U_B3 = np.zeros((16, 16), dtype=float)
    V_B3 = np.zeros((16, 16), dtype=float)
    W_B3 = np.zeros((16, 16), dtype=float)
    
    for i in range(4):
        for k in range(4):
            r = i * 4 + k
            U_B3[i * 4 + k, r] = 1 # Wait, U is n*m = 16. Index is i*m+j. Here m=4.
            # A is 4x4. U index is i*4+j.
            # So U_B3[i*4+k, r] = 1.
            V_B3[k * 4 + 8, r] = 1 # B is 4x9. V index is j*p+k. Here j=k? No.
            # B3 is col 8. B[j, 8].
            # V index is j*p + k. Here k is the row index of B? No.
            # B is m x p. m=4, p=9.
            # B[j, k].
            # V index is j*p + k.
            # So for B3, k=8.
            # V[j, 8] = B[j, 8].
            # So V_B3[j*9+8, r] = 1.
            # In my loop, j is k from 0..3.
            V_B3[k * 9 + 8, r] = 1
            W_B3[8 * 4 + i, r] = 1 # W index is k*n+i. Here k=8, i is row of C.
            # W[8*4+i, r] = 1.
    
    U[:, 98:114] = U_B3
    V[:, 98:114] = V_B3
    W[:, 98:114] = W_B3
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
