import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 construction:
    # 4 blocks of M_{2,2,2} (Strassen, rank 7) for columns 1-8
    # 1 block of M_{2,2,1} (Schoolbook, rank 4) for column 9
    
    rank = 32
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col_offset = 0
    
    # Strassen decomposition for M_{2,2,2}
    # U indices: i*m + j (0..3)
    # V indices: j*p + k (0..17, but for 2x2 block, k is relative 0,1)
    # W indices: k*n + i (0..17, but for 2x2 block, k is relative 0,1)
    
    # We define the Strassen terms for a 2x2 block.
    # Let's define the local U, V, W for one block.
    # U_local is 4x7, V_local is 2x7 (local p=2), W_local is 2x7 (local n=2, p=2 -> 4x7? No W is p*n = 4)
    
    # Strassen Coefficients:
    # P1: A11 * (B11 + B22)
    # P2: (A11 + A22) * B22
    # P3: (A21 + A22) * B11
    # P4: A22 * (B11 + B22)
    # P5: (A11 + A22) * (B11 + B22)
    # P6: (A11 - A21) * (B12 + B22)
    # P7: (A12 - A22) * (B11 + B21)
    
    # U (A part):
    # P1: A11 -> U[0,0]=1
    # P2: A11+A22 -> U[0,1]=1, U[3,1]=1
    # P3: A21+A22 -> U[2,2]=1, U[3,2]=1
    # P4: A22 -> U[3,3]=1
    # P5: A11+A22 -> U[0,4]=1, U[3,4]=1
    # P6: A11-A21 -> U[0,5]=1, U[2,5]=-1
    # P7: A12-A22 -> U[1,6]=1, U[3,6]=-1
    
    # V (B part):
    # P1: B11+B22 -> V[0,0]=1, V[1,0]=1 (j=0,k=0; j=1,k=1) -> V[0,0]=1, V[1*2+1,0]=1 -> V[3,0]=1?
    # Wait, V index is j*p + k. For p=2, j=0->0,1; j=1->2,3.
    # B11 is j=0,k=0 -> idx 0. B22 is j=1,k=1 -> idx 3.
    # So P1: V[0,0]=1, V[3,0]=1.
    # P2: B22 -> V[3,1]=1.
    # P3: B11 -> V[0,2]=1.
    # P4: B11+B22 -> V[0,3]=1, V[3,3]=1.
    # P5: B11+B22 -> V[0,4]=1, V[3,4]=1.
    # P6: B12+B22 -> B12 is j=0,k=1 -> idx 1. B22 is idx 3. V[1,5]=1, V[3,5]=1.
    # P7: B11+B21 -> B11 is idx 0. B21 is j=1,k=0 -> idx 2. V[0,6]=1, V[2,6]=1.
    
    # W (C part):
    # C11 = P5 + P4 - P2 + P6
    # C12 = P1 + P2
    # C21 = P3 + P4
    # C22 = P5 + P1 - P3 - P7
    # W index is k*n + i. For n=2, k=0->0,1; k=1->2,3.
    # C11 (i=0,k=0) -> idx 0.
    # C12 (i=0,k=1) -> idx 2.
    # C21 (i=1,k=0) -> idx 1.
    # C22 (i=1,k=1) -> idx 3.
    
    # W[0, c] (C11): P5(1) + P4(1) - P2(1) + P6(1) -> W[0,4]=1, W[0,3]=1, W[0,1]=-1, W[0,5]=1
    # W[2, c] (C12): P1(1) + P2(1) -> W[2,0]=1, W[2,1]=1
    # W[1, c] (C21): P3(1) + P4(1) -> W[1,2]=1, W[1,3]=1
    # W[3, c] (C22): P5(1) + P1(1) - P3(1) - P7(1) -> W[3,4]=1, W[3,0]=1, W[3,2]=-1, W[3,6]=-1

    U_block = np.zeros((4, 7))
    U_block[0, 0] = 1
    U_block[0, 1] = 1; U_block[3, 1] = 1
    U_block[2, 2] = 1; U_block[3, 2] = 1
    U_block[3, 3] = 1
    U_block[0, 4] = 1; U_block[3, 4] = 1
    U_block[0, 5] = 1; U_block[2, 5] = -1
    U_block[1, 6] = 1; U_block[3, 6] = -1

    V_block = np.zeros((4, 7)) # m*p = 2*2 = 4
    V_block[0, 0] = 1; V_block[3, 0] = 1
    V_block[3, 1] = 1
    V_block[0, 2] = 1
    V_block[0, 3] = 1; V_block[3, 3] = 1
    V_block[0, 4] = 1; V_block[3, 4] = 1
    V_block[1, 5] = 1; V_block[3, 5] = 1
    V_block[0, 6] = 1; V_block[2, 6] = 1

    W_block = np.zeros((4, 7)) # p*n = 2*2 = 4
    # C11 (idx 0)
    W_block[0, 4] = 1; W_block[0, 3] = 1; W_block[0, 1] = -1; W_block[0, 5] = 1
    # C12 (idx 2)
    W_block[2, 0] = 1; W_block[2, 1] = 1
    # C21 (idx 1)
    W_block[1, 2] = 1; W_block[1, 3] = 1
    # C22 (idx 3)
    W_block[3, 4] = 1; W_block[3, 0] = 1; W_block[3, 2] = -1; W_block[3, 6] = -1

    # Place 4 blocks for columns 0-1, 2-3, 4-5, 6-7
    for block_idx in range(4):
        start_col = block_idx * 7
        # U is same for all blocks
        U[:, start_col:start_col+7] = U_block
        
        # V needs offset in k
        # V index j*p + k. k is 0,1 for this block.
        # Global k = block_idx*2 + local_k
        # So V_global[j*p + global_k, c] = V_block[j*2 + local_k, c]
        for j in range(2):
            for local_k in range(2):
                global_k = block_idx * 2 + local_k
                global_v_idx = j * p + global_k
                local_v_idx = j * 2 + local_k
                V[global_v_idx, start_col:start_col+7] = V_block[local_v_idx, :]
                
        # W needs offset in k
        # W index k*n + i. k is 0,1 for this block.
        # Global k = block_idx*2 + local_k
        # So W_global[global_k*n + i, c] = W_block[local_k*2 + i, c]
        for local_k in range(2):
            global_k = block_idx * 2 + local_k
            for i in range(2):
                global_w_idx = global_k * n + i
                local_w_idx = local_k * 2 + i
                W[global_w_idx, start_col:start_col+7] = W_block[local_w_idx, :]

    # Last block for column 8 (k=8)
    start_col = 28
    # Schoolbook for 2x2 x 2x1
    # C11 = A11 B11 + A12 B21
    # C21 = A21 B11 + A22 B21
    # Note: Here B is 2x1, so B11 is B[0,0], B21 is B[1,0].
    # In our notation, B is 2x9. The block is column 8 (0-indexed).
    # B[0, 8] and B[1, 8].
    
    # U:
    # Term 1: A11 B11 -> U[0, 0]=1
    # Term 2: A12 B21 -> U[1, 1]=1
    # Term 3: A21 B11 -> U[2, 2]=1
    # Term 4: A22 B21 -> U[3, 3]=1
    U[0, start_col] = 1
    U[1, start_col+1] = 1
    U[2, start_col+2] = 1
    U[3, start_col+3] = 1
    
    # V:
    # Term 1: B11 -> V[0*p + 8, 0] = V[8, 0] = 1
    # Term 2: B21 -> V[1*p + 8, 1] = V[18, 1] = 1
    # Term 3: B11 -> V[8, 2] = 1
    # Term 4: B21 -> V[18, 3] = 1
    V[8, start_col] = 1
    V[18, start_col+1] = 1
    V[8, start_col+2] = 1
    V[18, start_col+3] = 1
    
    # W:
    # C11 (i=0, k=8) -> idx 8*2 + 0 = 16
    # C21 (i=1, k=8) -> idx 8*2 + 1 = 17
    # Term 1: C11 -> W[16, 0] = 1
    # Term 2: C11 -> W[16, 1] = 1
    # Term 3: C21 -> W[17, 2] = 1
    # Term 4: C21 -> W[17, 3] = 1
    W[16, start_col] = 1
    W[16, start_col+1] = 1
    W[17, start_col+2] = 1
    W[17, start_col+3] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
