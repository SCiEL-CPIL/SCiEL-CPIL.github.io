import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def get_strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def get_3x3x3_decomp():
    # Returns rank 23 decomposition for 3x3x3
    # Using a known set of 23 triples
    # If not available, fallback to 27
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Hardcoded Rank 23 decomposition
    # This is a simplified representation for the purpose of this task.
    # In a real scenario, one would use the full 23 vectors from Landsberg.
    # Here I will use a placeholder that is valid if I can construct it.
    # Since I cannot guarantee the 23 vectors, I will use the schoolbook 27 
    # but map it to the 23 slots? No, that's invalid.
    
    # I will use the schoolbook 27 decomposition but pad to 23? No.
    # I will return None and let the caller handle fallback.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_indices, v_indices, w_indices, U_block, V_block, W_block):
        rank = U_block.shape[1]
        for r in range(rank):
            U_vec = np.zeros(n * m)
            V_vec = np.zeros(m * p)
            W_vec = np.zeros(p * n)
            
            for i, idx in enumerate(u_indices):
                U_vec[idx] = U_block[i, r]
            for i, idx in enumerate(v_indices):
                V_vec[idx] = V_block[i, r]
            for i, idx in enumerate(w_indices):
                W_vec[idx] = W_block[i, r]
                
            U_parts.append(U_vec)
            V_parts.append(V_vec)
            W_parts.append(W_vec)

    # Split:
    # A: 3x4 -> A1 (3x3), A2 (3x1)
    # B: 4x5 -> B1 (3x3), B2 (3x2), B3 (1x3), B4 (1x2)
    # C: 3x5
    
    # Block 1: A1 (3x3) * B1 (3x3) -> C1 (3x3)
    # Use Rank 23 if available, else 27
    decomp_3x3x3 = get_3x3x3_decomp()
    if decomp_3x3x3 is None:
        # Fallback to schoolbook 27
        U1, V1, W1 = np.zeros((9, 27)), np.zeros((9, 27)), np.zeros((9, 27))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i*9 + j*3 + k
                    U1[i*3+j, r] = 1
                    V1[j*3+k, r] = 1
                    W1[k*3+i, r] = 1
        rank1 = 27
    else:
        U1, V1, W1 = decomp_3x3x3
        rank1 = 23

    u_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8] # A1 is 3x3, indices 0..8 in A (3x4)
    v_idx = [0, 1, 2, 6, 7, 8, 12, 13, 14] # B1 is 3x3, rows 0..2, cols 0..2 in B (4x5)
    w_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8] # C1 is 3x3, cols 0..2 in C (3x5)
    
    add_block(u_idx, v_idx, w_idx, U1, V1, W1)
    
    # Block 2: A1 (3x3) * B2 (3x2) -> C2 (3x2)
    # B2 is rows 0..2, cols 3..4 in B.
    # Split into two 3x3x1 problems? Or one 3x3x2?
    # Rank of 3x3x2 is 18 (9+9).
    # We can use two 3x3x1 decompositions.
    # 3x3x1 schoolbook rank 9.
    U2a, V2a, W2a = np.zeros((9, 9)), np.zeros((9, 9)), np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            r = i*3 + j
            U2a[i*3+j, r] = 1
            V2a[j, r] = 1 # B is 3x1, so index j
            W2a[i, r] = 1 # C is 3x1, index i (col 3 of C)
            
    U2b, V2b, W2b = np.zeros((9, 9)), np.zeros((9, 9)), np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            r = i*3 + j
            U2b[i*3+j, r] = 1
            V2b[j, r] = 1 # B is 3x1, so index j
            W2b[i, r] = 1 # C is 3x1, index i (col 4 of C)

    # Map Block 2a (col 3)
    u_idx2 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    v_idx2a = [0, 1, 2, 6, 7, 8, 12, 13, 14] # Wait, B2 is 3x2.
    # B indices in global 4x5:
    # Row 0: 0,1,2,3,4
    # Row 1: 5,6,7,8,9
    # Row 2: 10,11,12,13,14
    # B2 is rows 0..2, cols 3..4.
    # Indices: 3,4, 8,9, 13,14.
    # We split into col 3 (indices 3,8,13) and col 4 (indices 4,9,14).
    
    v_idx2a = [3, 8, 13] # B2 col 0 (global col 3)
    w_idx2a = [9, 10, 11] # C2 col 0 (global col 3). C index k*n+i. k=3. 3*3+0=9, 10, 11.
    add_block(u_idx2, v_idx2a, w_idx2a, U2a, V2a, W2a)
    
    v_idx2b = [4, 9, 14] # B2 col 1 (global col 4)
    w_idx2b = [12, 13, 14] # C2 col 1 (global col 4). k=4. 4*3+0=12, 13, 14.
    add_block(u_idx2, v_idx2b, w_idx2b, U2b, V2b, W2b)

    # Block 3: A2 (3x1) * B3 (1x3) -> C3 (3x3)
    # A2 is col 3 of A. Indices: 3, 7, 11.
    # B3 is row 3 of B. Indices: 15, 16, 17, 18, 19.
    # Wait, B is 4x5.
    # Row 3: 15, 16, 17, 18, 19.
    # B3 is 1x3, so cols 0..2 of B3? No, B3 corresponds to the part of B that multiplies A2.
    # A = [A1, A2]. A2 is 3x1.
    # B = [B1, B2; B3, B4].
    # B3 is row 3, cols 0..2. Indices: 15, 16, 17.
    # B4 is row 3, cols 3..4. Indices: 18, 19.
    
    # A2 indices: 3 (row 0), 7 (row 1), 11 (row 2).
    u_idx3 = [3, 7, 11]
    
    # B3 indices: 15, 16, 17.
    v_idx3 = [15, 16, 17]
    
    # C3 indices: C[0..2, 0..2].
    # C index k*n+i.
    # k=0: 0,1,2
    # k=1: 3,4,5
    # k=2: 6,7,8
    w_idx3 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    # Rank 9 schoolbook for 3x1x3.
    U3 = np.zeros((3, 9))
    V3 = np.zeros((3, 9))
    W3 = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            r = i*3 + j
            U3[i, r] = 1
            V3[j, r] = 1
            W3[j*3+i, r] = 1
            
    add_block(u_idx3, v_idx3, w_idx3, U3, V3, W3)

    # Block 4: A2 (3x1) * B4 (1x2) -> C4 (3x2)
    u_idx4 = [3, 7, 11]
    v_idx4 = [18, 19] # B4 indices
    w_idx4 = [9, 10, 11, 12, 13, 14] # C4 indices (cols 3,4)
    
    # Rank 6 schoolbook for 3x1x2.
    U4 = np.zeros((3, 6))
    V4 = np.zeros((2, 6))
    W4 = np.zeros((6, 6))
    for i in range(3):
        for j in range(2):
            r = i*2 + j
            U4[i, r] = 1
            V4[j, r] = 1
            W4[j*3+i, r] = 1
            
    add_block(u_idx4, v_idx4, w_idx4, U4, V4, W4)

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
