import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 8.0
    
    n, m, p = 4, 4, 4
    
    # Laderman's algorithm for 2x2 matrix multiplication (rank 7)
    # M1 = A00 * (B00 + B01)
    # M2 = (A01 - A00) * (B00 + B01)
    # M3 = (A00 + A11) * B11
    # M4 = A11 * (B01 - B11)
    # M5 = (A10 + A00) * B00
    # M6 = (A10 + A00) * (B11 - B00)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Build factor matrices for 2x2 Laderman
    U2 = np.zeros((4, 7), dtype=np.int64)
    V2 = np.zeros((4, 7), dtype=np.int64)
    W2 = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = A00 * (B00 + B01)
    U2[0, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = 1
    
    # M2 = (A01 - A00) * (B00 + B01)
    U2[1, 1] = 1; U2[0, 1] = -1
    V2[0, 1] = 1; V2[1, 1] = 1
    W2[0, 1] = -1; W2[1, 1] = -1
    
    # M3 = (A00 + A11) * B11
    U2[0, 2] = 1; U2[3, 2] = 1
    V2[3, 2] = 1
    W2[0, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11 * (B01 - B11)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[3, 3] = -1
    W2[3, 3] = 1
    
    # M5 = (A10 + A00) * B00
    U2[2, 4] = 1; U2[0, 4] = 1
    V2[0, 4] = 1
    W2[2, 4] = 1; W2[3, 4] = 1
    
    # M6 = (A10 + A00) * (B11 - B00)
    U2[2, 5] = 1; U2[0, 5] = 1
    V2[3, 5] = 1; V2[0, 5] = -1
    W2[2, 5] = -1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[1, 6] = 1
    
    # Verify the 2x2 decomposition works
    # For each output element C_ij, sum over k of U[i,k] * V[j,k] * W[output,k] should give C_ij
    
    # Now compute Kronecker products for 4x4
    # The 4x4 matrix can be viewed as 2x2 blocks of 2x2 matrices
    # So we need to compose the 2x2 algorithm with itself
    
    K_U = np.kron(U2, U2)
    K_V = np.kron(V2, V2)
    K_W = np.kron(W2, W2)
    
    # K_U has shape (16, 49), K_V has shape (16, 49), K_W has shape (16, 49)
    # We need to permute rows to match the required output ordering
    
    # For U and V: row index corresponds to input position
    # Original 2x2: positions 0,1,2,3 correspond to (0,0),(0,1),(1,0),(1,1)
    # Kron: position (a,b) where a,b in {0,1,2,3}
    # a encodes the block row/col within the 2x2 block structure
    # b encodes the position within the 2x2 sub-block
    
    # For 4x4, position (i,j) where i,j in {0,1,2,3}
    # Block row = i//2, block col = j//2
    # Within block: row = i%2, col = j%2
    
    # So for U (left input A), position (i,j):
    #   block index = (i//2)*2 + (j//2)  -- this is the "outer" 2x2 position
    #   inner index = (i%2)*2 + (j%2)    -- this is the "inner" 2x2 position
    #   kron index = block_index * 4 + inner_index
    
    # For V (right input B), same mapping
    
    # For W (output C), position (k,l) where k=row, l=col of C
    # But the trace convention is W[k*n+i, r] with n=4, so output index = k*4 + l
    
    perm_UV = np.zeros(16, dtype=np.int64)
    perm_W = np.zeros(16, dtype=np.int64)
    
    for i in range(4):
        for j in range(4):
            out_idx = i * 4 + j
            
            # For U and V:
            block_r = i // 2
            block_c = j // 2
            outer = block_r * 2 + block_c
            inner_r = i % 2
            inner_c = j % 2
            inner = inner_r * 2 + inner_c
            kron_idx = outer * 4 + inner
            
            perm_UV[out_idx] = kron_idx
            
            # For W: output C[k,l], but we're mapping from (i,j) to output position
            # The output element at position (i,j) in C comes from the product
            # So for W, the row index in the final W matrix corresponds to output position
            # W[k*n+i, r] where k is col, i is row in the trace convention
            # But for the tensor decomposition, we just need to match the kron index
            
            # For W, the output position (i,j) maps to:
            #   block_r = i // 2 (row block)
            #   block_c = j // 2 (col block)
            #   outer = block_r * 2 + block_c
            #   inner_r = i % 2
            #   inner_c = j % 2
            #   inner = inner_r * 2 + inner_c
            #   kron_idx = outer * 4 + inner
            # But the output is ordered as k*4 + l where k=col, l=row
            # So for W, row index = j * 4 + i (col-major for the trace convention)
            
            w_out_idx = j * 4 + i
            perm_W[w_out_idx] = kron_idx
    
    U_arr = K_U[perm_UV]
    V_arr = K_V[perm_UV]
    W_arr = K_W[perm_W]
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}
    
    return {
        "U": U_arr.tolist(),
        "V": V_arr.tolist(),
        "W": W_arr.tolist()
    }
