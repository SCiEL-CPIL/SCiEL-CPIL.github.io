"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)] # 4 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 16 x 28
    
    # Strassen linear forms for A (2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    A_forms = [
        [1, 0, 0, 1], # L1: a11 + a22
        [0, 0, 1, 1], # L2: a21 + a22
        [1, 0, 0, 0], # L3: a11
        [0, 0, 0, 1], # L4: a22
        [1, 1, 0, 0], # L5: a11 + a12
        [-1, 0, 1, 0], # L6: a21 - a11
        [0, 1, 0, -1]  # L7: a12 - a22
    ]
    
    # For each of the 4 blocks, we define the B and W forms
    # Block 0: B11 (j=0, k=0,1)
    # Block 1: B12 (j=0, k=2,3)
    # Block 2: B21 (j=1, k=4,5)
    # Block 3: B22 (j=1, k=6,7)
    
    # B indices for each block: [b11, b12, b21, b22] in global V indices
    # V index = j*p + k = j*8 + k
    block_B_indices = [
        [0, 1, 8, 9],   # Block 0: j=0, k=0,1 -> V[0], V[1], V[8], V[9]
        [2, 3, 10, 11], # Block 1: j=0, k=2,3 -> V[2], V[3], V[10], V[11]
        [4, 5, 12, 13], # Block 2: j=1, k=4,5 -> V[4], V[5], V[12], V[13]
        [6, 7, 14, 15]  # Block 3: j=1, k=6,7 -> V[6], V[7], V[14], V[15]
    ]
    
    # W indices for each block: [c11, c12, c21, c22] in global W indices
    # W index = k*n + i = k*2 + i
    # c11: i=0, k=0 -> W[0]
    # c12: i=0, k=1 -> W[2]
    # c21: i=1, k=0 -> W[1]
    # c22: i=1, k=1 -> W[3]
    block_W_indices = [
        [0, 2, 1, 3],   # Block 0: k=0,1
        [4, 6, 5, 7],   # Block 1: k=2,3 -> W[4], W[6], W[5], W[7]
        [8, 10, 9, 11], # Block 2: k=4,5 -> W[8], W[10], W[9], W[11]
        [12, 14, 13, 15] # Block 3: k=6,7 -> W[12], W[14], W[13], W[15]
    ]
    
    # Strassen linear forms for B (2x2)
    # b11, b12, b21, b22
    B_forms = [
        [1, 0, 0, 1], # M1: b11 + b22
        [1, 0, 0, 0], # M2: b11
        [0, 1, 0, -1], # M3: b12 - b22
        [0, 0, 1, -1], # M4: b21 - b11
        [0, 0, 0, 1], # M5: b22
        [1, 1, 0, 0], # M6: b11 + b12
        [0, 0, 1, 1]  # M7: b21 + b22
    ]
    
    # Strassen linear forms for C (2x2)
    # c11, c12, c21, c22
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    C_forms = [
        [1, 0, 0, 1, -1, 0, 1], # c11
        [0, 1, 0, 0, 1, 0, 0], # c12
        [0, 0, 1, 1, 0, 0, 0], # c21
        [1, -1, 1, 0, 0, 1, 0] # c22
    ]
    
    for b in range(4):
        base_col = b * 7
        B_idx = block_B_indices[b]
        W_idx = block_W_indices[b]
        
        for t in range(7):
            col = base_col + t
            
            # Fill U
            for i in range(4):
                U[i][col] = A_forms[t][i]
                
            # Fill V
            # B_forms[t] has 4 entries corresponding to b11, b12, b21, b22
            for k in range(4):
                row = B_idx[k]
                V[row][col] = B_forms[t][k]
                
            # Fill W
            # C_forms[k] has 7 entries corresponding to M1..M7
            # We are filling column t, so we take the t-th element of C_forms[k]
            for k in range(4):
                row = W_idx[k]
                W[row][col] = C_forms[k][t]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
