import numpy as np

def get_strassen_2x2():
    # Returns U, V, W for 2x2x2 multiplication
    # U: 4x7, V: 4x7, W: 4x7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1 # C00
    W[3, 0] += 1 # C11
    
    # M2 = (A10 + A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1 # C10
    
    # M3 = A00(B01 - B11)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1 # C01
    W[3, 2] += 1 # C11
    
    # M4 = A11(B10 - B00)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1 # C00
    W[2, 3] += 1 # C10
    
    # M5 = (A00 + A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1 # C00
    W[1, 4] += 1 # C01
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1 # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1 # C00
    W[1, 6] += 1 # C01
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U2, V2, W2 = get_strassen_2x2()
    
    # Build 4x4x4 factors (Rank 49)
    # We need to map 2x2 blocks to 4x4 indices.
    # U4 indices: i in 0..3, j in 0..3.
    # Block (bi, bj) of A corresponds to i = 2*bi + i2, j = 2*bj + j2.
    # U2 indices: i2, j2.
    # So U4[2*bi + i2, 2*bj + j2, c4] = U2[i2, j2, c2]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # Wait, Strassen combines blocks.
                # The rank 49 comes from 7 block multiplications, each of rank 7.
                # The block multiplication is for the 2x2 block matrix.
                # Let A_blocks be 2x2 matrix of 2x2 blocks.
                # M_block = A_blocks * B_blocks.
                # Each M_block element is a 2x2 matrix product.
                # So for each of the 7 Strassen terms on the block level,
                # we perform a 2x2 matrix multiplication.
                
                # Block index in Strassen:
                # The 7 terms correspond to specific linear combos of blocks.
                # We need to map the "block" indices to the "scalar" indices.
                
                # Let's iterate through the 7 block terms.
                # For each block term t (0..6), we have a 2x2 multiplication.
                # This contributes 7 scalar multiplications.
                # Total 49.
                
                # We need the U2, V2, W2 for the scalar multiplication.
                # But the U, V, W for the block multiplication define the linear combos.
                
                # Let's define the block-level Strassen factors.
                # U_block (4x7), V_block (4x7), W_block (4x7)
                # U_block indices: (bi, bj)
                # V_block indices: (bj, bk)
                # W_block indices: (bk, bi)
                
                # We can reuse the get_strassen_2x2 logic for the blocks?
                # Yes, the structure is identical.
                
                # So for each block term t (0..6):
                #   For each scalar term s (0..6) in the 2x2 multiplication:
                #       col = t * 7 + s
                #       U4[2*bi + i2, 2*bj + j2, col] += U_block[bi, bj, t] * U2[i2, j2, s]
                #       V4[2*bj + j2, 2*bk + k2, col] += V_block[bj, bk, t] * V2[j2, k2, s]
                #       W4[2*bk + k2, 2*bi + i2, col] += W_block[bk, bi, t] * W2[k2, i2, s]
                
                pass # Placeholder

    # Actually, it's easier to just generate the 4x4 factors directly using the block structure.
    # I will implement the loop above properly in the final code.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
