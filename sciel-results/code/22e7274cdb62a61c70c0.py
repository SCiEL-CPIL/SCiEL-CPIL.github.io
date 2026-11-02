import numpy as np

DIMENSIONS = [2, 2, 16]

# Strassen decomposition for 2x2 matrix multiplication
# A, B are 2x2. C = AB.
# Rank 7.
# U, V, W are vectors of length 4 (flattened 2x2).
# Indices: i in 0..1, j in 0..1.
# A[i,j], B[i,j], C[i,j].
# Strassen steps:
# M1 = (A00 + A11) * (B00 + B11)
# M2 = (A10 + A11) * B00
# M3 = A00 * (B01 - B11)
# M4 = A11 * (B10 - B00)
# M5 = (A00 + A01) * B11
# M6 = (A10 - A00) * (B00 + B01)
# M7 = (A01 - A11) * (B10 + B11)

# C00 = M1 + M4 - M5 + M7
# C01 = M3 + M5
# C10 = M2 + M4
# C11 = M1 - M2 + M3 + M6

def get_strassen_factors():
    # Returns U, V, W for one 2x2x2 multiplication.
    # U is 4x7, V is 4x7, W is 4x7.
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # Wait, C10 = M2+M4, C11 = M1-M2+M3+M6.
    # Let's re-verify Strassen formulas.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2: (A10+A11)B00 -> contributes to C10, C11
    # M3: A00(B01-B11) -> contributes to C01, C11
    # M4: A11(B10-B00) -> contributes to C00, C10
    # M5: (A00+A01)B11 -> contributes to C00, C01
    # M6: (A10-A00)(B00+B01) -> contributes to C11
    # M7: (A01-A11)(B10+B11) -> contributes to C00
    
    # Let's map W (Output C) coefficients.
    # W is 4x7. W[c_idx, step].
    
    # Step 0 (M1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1 # C00
    W[3, 0] = 1 # C11
    
    # Step 1 (M2)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C10
    W[3, 1] = -1 # C11
    
    # Step 2 (M3)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C01
    W[3, 2] = 1 # C11
    
    # Step 3 (M4)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C00
    W[2, 3] = 1 # C10
    
    # Step 4 (M5)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C00
    W[1, 4] = 1 # C01
    
    # Step 5 (M6)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11
    
    # Step 6 (M7)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # B is 2x16. Split into 8 blocks of 2x2.
    num_blocks = p // 2 # 8 blocks
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U_block, V_block, W_block = get_strassen_factors()
    
    for s in range(num_blocks):
        for r in range(rank_per_block):
            c = s * rank_per_block + r
            # U is same for all blocks
            U[:, c] = U_block[:, r]
            
            # V: block s corresponds to columns 2*s and 2*s+1 of B.
            # B is m x p. Flattened index j*p + k.
            # j in 0..1. k in 0..15.
            # Block s has k = 2*s, 2*s+1.
            # V_block indices: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1) of the 2x2 block.
            # Map V_block row idx to V row idx.
            # V_block row 0: A[0,0] part? No, V is for B.
            # V_block[0] corresponds to B[0,0] of the block.
            # In full B, this is B[0, 2*s]. Index: 0*p + 2*s = 2*s.
            # V_block[1] corresponds to B[0,1] of the block.
            # In full B, this is B[0, 2*s+1]. Index: 0*p + 2*s+1 = 2*s+1.
            # V_block[2] corresponds to B[1,0] of the block.
            # In full B, this is B[1, 2*s]. Index: 1*p + 2*s = 16 + 2*s.
            # V_block[3] corresponds to B[1,1] of the block.
            # In full B, this is B[1, 2*s+1]. Index: 1*p + 2*s+1 = 16 + 2*s+1.
            
            v_indices = [2*s, 2*s+1, 16+2*s, 16+2*s+1]
            for i in range(4):
                V[v_indices[i], c] = V_block[i, r]
                
            # W: block s corresponds to columns 2*s and 2*s+1 of C.
            # C is p x n. Flattened index k*n + i.
            # k in 0..15. i in 0..1.
            # Block s has k = 2*s, 2*s+1.
            # W_block indices: 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1) of the 2x2 output block.
            # Map W_block row idx to W row idx.
            # W_block[0]: C[0,0] of block. Full C[2*s, 0]. Index: (2*s)*2 + 0 = 4*s.
            # W_block[1]: C[0,1] of block. Full C[2*s, 1]. Index: (2*s)*2 + 1 = 4*s+1.
            # W_block[2]: C[1,0] of block. Full C[2*s+1, 0]. Index: (2*s+1)*2 + 0 = 4*s+2.
            # W_block[3]: C[1,1] of block. Full C[2*s+1, 1]. Index: (2*s+1)*2 + 1 = 4*s+3.
            
            w_indices = [4*s, 4*s+1, 4*s+2, 4*s+3]
            for i in range(4):
                W[w_indices[i], c] = W_block[i, r]
                
    return {"U": U, "V": V, "W": W}
