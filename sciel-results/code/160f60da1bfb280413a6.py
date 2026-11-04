import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank for 4x4x4 using Strassen is 49
    # Rank for 4x4x3 using Schoolbook is 48
    rank_444 = 49
    rank_443 = 48
    total_rank = rank_444 + rank_443
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1: 4x4x4 (First 4 columns of B)
    # We need to fill U[:, 0:49], V[:, 0:49], W[:, 0:49]
    # This is complex to hardcode directly. 
    # Alternative: Just use Schoolbook for 4x4x4? Rank 64. Total 112. No.
    # I will implement the Strassen 4x4x4 generation.
    
    # Strassen 2x2x2 base tensor
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen coefficients for 2x2
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # U2[i, k] is coeff of A[i] in Pk
    # V2[j, k] is coeff of B[j] in Pk
    # W2[l, k] is coeff of C[l] in Pk (actually W is for output, so W[l, k] is coeff of Pk in C[l])
    
    # P1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = 0; W2[2, 0] = 0; W2[3, 0] = 1 # C00+=P1, C11+=P1
    
    # P2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C10+=P2, C11-=P2
    
    # P3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1 # C01+=P3, C11+=P3
    
    # P4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1 # C00+=P4, C10+=P4
    
    # P5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1 # C00-=P5, C01+=P5
    
    # P6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C11+=P6
    
    # P7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = -1 # C00+=P7, C11-=P7
    
    # Now build 4x4x4 Strassen
    # A is 2x2 blocks of 2x2.
    # A_blocks: A00, A01, A10, A11
    # Each A_block is 2x2.
    # U4 will be 16x49.
    # The 49 multiplications are 7 groups of 7 (one group per Pk).
    # For each Pk, we use U2, V2, W2 but mapped to the 2x2 blocks.
    
    # Block indices: 0->00, 1->01, 2->10, 3->11
    # U4[i, col] where i is 0..15 (4x4 index), col is 0..48.
    # col = k * 7 + sub_col
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49)) # V is for B (4x4), so 16x49
    W4 = np.zeros((16, 49)) # W is for C (4x4), so 16x49
    
    # Mapping of block indices to 4x4 indices
    # Block 00: rows 0,1 cols 0,1 -> indices 0,1,4,5
    # Block 01: rows 0,1 cols 2,3 -> indices 2,3,6,7
    # Block 10: rows 2,3 cols 0,1 -> indices 8,9,12,13
    # Block 11: rows 2,3 cols 2,3 -> indices 10,11,14,15
    
    block_indices = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    for k in range(7):
        base_col = k * 7
        for sub_col in range(7):
            col = base_col + sub_col
            # U part
            for block in range(4):
                coeff = U2[block, sub_col]
                if coeff != 0:
                    for idx in block_indices[block]:
                        U4[idx, col] += coeff
            # V part
            for block in range(4):
                coeff = V2[block, sub_col]
                if coeff != 0:
                    for idx in block_indices[block]:
                        V4[idx, col] += coeff
            # W part
            for block in range(4):
                coeff = W2[block, sub_col]
                if coeff != 0:
                    for idx in block_indices[block]:
                        W4[idx, col] += coeff
                        
    # Fill U, V, W for first 49 cols
    U[:, 0:49] = U4
    V[:, 0:49] = V4
    W[:, 0:49] = W4
    
    # Part 2: 4x4x3 (Last 3 columns of B)
    # Schoolbook
    # n=4, m=4, p=3
    col_offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                col = col_offset + (i * 4 + j) * 3 + k
                U[i * 4 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[k * 4 + i, col] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
