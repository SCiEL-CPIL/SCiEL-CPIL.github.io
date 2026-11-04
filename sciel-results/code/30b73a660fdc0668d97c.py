import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 56
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 decomposition
    # U, V, W are 4x7
    # Indices: i,j,k in {0,1}
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    # r=0
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    # Extract Strassen blocks
    S_U = U[:, :7].copy()
    S_V = V[:, :7].copy()
    S_W = W[:, :7].copy()
    
    # We need to place 8 blocks of 2x2x2 into the 2x4x8 structure.
    # The structure is:
    # A (2x4) * B (4x8)
    # Split A into A_L (2x2, j=0,1) and A_R (2x2, j=2,3)
    # Split B into 4 blocks of 2x2 in k: B_0 (k=0,1), B_1 (k=2,3), B_2 (k=4,5), B_3 (k=6,7)
    # We need to compute A_L * B_0, A_L * B_1, A_L * B_2, A_L * B_3
    # and A_R * B_0, A_R * B_1, A_R * B_2, A_R * B_3
    
    # Helper to place a Strassen block
    def place_block(rank_offset, i_off, j_off, k_off):
        # i_off: offset for i (0 or 1)
        # j_off: offset for j (0, 1, 2, or 3)
        # k_off: offset for k (0, 2, 4, or 6)
        
        # U indices: (i + i_off) * 4 + (j + j_off)
        # S_U indices: i * 2 + j
        for r in range(7):
            rank_idx = rank_offset + r
            for i in range(2):
                for j in range(2):
                    val = S_U[i*2+j, r]
                    if val != 0:
                        U[(i+i_off)*4 + (j+j_off), rank_idx] += val
                        
        # V indices: (j + j_off) * 8 + (k + k_off)
        # S_V indices: j * 2 + k
        for r in range(7):
            rank_idx = rank_offset + r
            for j in range(2):
                for k in range(2):
                    val = S_V[j*2+k, r]
                    if val != 0:
                        V[(j+j_off)*8 + (k+k_off), rank_idx] += val
                        
        # W indices: (k + k_off) * 2 + (i + i_off)
        # S_W indices: k * 2 + i
        for r in range(7):
            rank_idx = rank_offset + r
            for k in range(2):
                for i in range(2):
                    val = S_W[k*2+i, r]
                    if val != 0:
                        W[(k+k_off)*2 + (i+i_off), rank_idx] += val

    # Place 8 blocks
    # Block 0: A_L (j=0), B_0 (k=0)
    place_block(0, 0, 0, 0)
    # Block 1: A_L (j=0), B_1 (k=2)
    place_block(7, 0, 0, 2)
    # Block 2: A_L (j=0), B_2 (k=4)
    place_block(14, 0, 0, 4)
    # Block 3: A_L (j=0), B_3 (k=6)
    place_block(21, 0, 0, 6)
    
    # Block 4: A_R (j=2), B_0 (k=0)
    place_block(28, 0, 2, 0)
    # Block 5: A_R (j=2), B_1 (k=2)
    place_block(35, 0, 2, 2)
    # Block 6: A_R (j=2), B_2 (k=4)
    place_block(42, 0, 2, 4)
    # Block 7: A_R (j=2), B_3 (k=6)
    place_block(49, 0, 2, 6)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
