"""Matrix multiplication 4,5,5 rank-minimal decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen 4x4 decomposition (Rank 49)
    # 2x2 Strassen Tensors
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Indices in 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Terms:
    # 1: (A11+A22)(B11+B22)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1; W2[1, 0] = 1; W2[2, 0] = 1 # C11+=M1, C22+=M1
    
    # 2: (A21+A22)B11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1 # C21+=M2, C22-=M2
    
    # 3: A11(B12-B22)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C12+=M3, C22+=M3
    
    # 4: A22(B21-B11)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C11+=M4, C21+=M4
    
    # 5: (A11+A12)B22
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[0, 4] = -1 # C12+=M5, C11-=M5
    
    # 6: (A21-A11)(B11+B12)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C22+=M6
    
    # 7: (A12-A22)(B21+B22)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C11+=M7

    # Kronecker product to get 4x4 (Rank 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    # U4 index i*4 + j
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            val_u = U2[i * 2 + i2, r1] * U2[j1 * 2 + j2, r2] # Wait, U2 index is i*2+j?
                            # U2 index for A is i*2+j.
                            # In 4x4, A index is i*4+j.
                            # i = 2*i1 + i2, j = 2*j1 + j2.
                            # U2 index for block (i1,j1) is i1*2 + i2? No.
                            # U2 is for 2x2 matrix. Index is row*2+col.
                            # For 4x4, we are combining two 2x2s.
                            # The U4 matrix corresponds to A (4x4).
                            # A is composed of 4 2x2 blocks.
                            # Block (i1, j1) of A corresponds to U2 indices.
                            # Actually, the Kronecker product of tensors T1 and T2:
                            # T = T1 x T2.
                            # U4[i1*4 + i2, j1*4 + j2]? No, U4 is flattened A.
                            # A index is i*4+j.
                            # i = 2*i1 + i2. j = 2*j1 + j2.
                            # The term r1 acts on the "row" part? No.
                            # Strassen is applied to the whole 4x4 matrix as a 2x2 block of 2x2s.
                            # So A = [[A11, A12], [A21, A22]] where Ai are 2x2.
                            # The Strassen terms M1..M7 are computed on Ai.
                            # So for each Strassen term r1, we have a linear combination of Ai.
                            # Then we apply Strassen again to the result?
                            # No, the rank is 49 because we have 7 terms for the outer 2x2, and each term involves 2x2 multiplications which cost 7 terms each?
                            # Yes.
                            # So for each outer term r1, we compute a 2x2 matrix product of linear combos.
                            # This inner product uses 7 terms r2.
                            # So the total tensor is sum over r1, r2 of (Tensor_r1 x Tensor_r2).
                            
                            # Let's verify the indexing.
                            # U4[i*4+j, r] should be U2[i_outer*2+i_inner, r1] * U2[j_outer*2+j_inner, r2]?
                            # No.
                            # A is 4x4.
                            # A[i,j] where i=2*i1+i2, j=2*j1+j2.
                            # The term r1 corresponds to a linear combo of the 2x2 blocks of A.
                            # The term r2 corresponds to a linear combo of the 2x2 blocks of B.
                            # The product is a 2x2 block of C.
                            # This seems correct.
                            
                            u_val = U2[i1*2 + i2, r1] * U2[j1*2 + j2, r2]
                            U4[i*4 + j, r] += u_val
                            
                            v_val = V2[i1*2 + i2, r1] * V2[j1*2 + j2, r2] # Wait, V2 indices for B?
                            # B index is j*4+k.
                            # j = 2*j1+j2, k = 2*k1+k2.
                            # V4[j*4+k, r] += V2[j1*2+j2, r1] * V2[k1*2+k2, r2]
                            # My loop only has i1,i2,j1,j2. I need k1,k2 for V.
                            
                            # Let's restructure the loop to be explicit.
    
    # Re-do Kronecker construction properly
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            U4[i*4 + j, r] += U2[i1*2 + i2, r1] * U2[j1*2 + j2, r2]
                            
                            for k1 in range(2):
                                for k2 in range(2):
                                    k = k1 * 2 + k2
                                    V4[j*4 + k, r] += V2[j1*2 + j2, r1] * V2[k1*2 + k2, r2]
                                    W4[k*4 + i, r] += W2[k1*2 + k2, r1] * W2[i1*2 + i2, r2]
                                    
    # Global Matrices
    R = 49 + 16 + 20
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Block 1: Strassen (Offset 0)
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U[i*5 + j, r] = U4[i*4 + j, r]
        for j in range(4):
            for k in range(4):
                V[j*5 + k, r] = V4[j*4 + k, r]
        for k in range(4):
            for i in range(4):
                W[k*4 + i, r] = W4[k*4 + i, r]
                
    # Block 2: Matrix-Vector (Offset 49)
    # A1 (4x4) * b12 (4x1) -> C_col4 (4x1)
    for t in range(16):
        r = 49 + t
        i = t // 4
        k = t % 4
        U[i*5 + k, r] = 1
        V[k*5 + 4, r] = 1
        W[4*4 + i, r] = 1
        
    # Block 3: Outer Product (Offset 65)
    # A2 (4x1) * B2 (1x5) -> C (4x5)
    for t in range(20):
        r = 65 + t
        i = t // 5
        k = t % 5
        U[i*5 + 4, r] = 1
        V[4*5 + k, r] = 1
        W[k*4 + i, r] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
