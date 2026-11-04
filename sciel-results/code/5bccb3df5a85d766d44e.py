"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    
    # Strassen 2x2 coefficients
    # Rows: 00, 01, 10, 11
    S_U = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, -1, -1]
    ])
    
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Lader 3x3 coefficients
    # Rows: 00, 01, 02, 10, 11, 12, 20, 21, 22
    L_U = np.array([
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0]
    ])
    
    L_V = L_U.copy()
    
    L_W = np.array([
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0]
    ])
    
    # Construct 6x6 decomposition
    # U_6 shape (36, 49)
    # V_6 shape (36, 49)
    # W_6 shape (36, 49)
    
    U_6 = np.zeros((36, 49))
    V_6 = np.zeros((36, 49))
    W_6 = np.zeros((36, 49))
    
    for k in range(7): # Strassen term
        for l in range(7): # Lader term
            col = k * 7 + l
            
            for b1 in range(2): # Block row
                for b2 in range(2): # Block col
                    for i in range(3): # Sub row
                        for j in range(3): # Sub col
                            # U index
                            idx_u = (b1 * 3 + i) * 6 + (b2 * 3 + j)
                            U_6[idx_u, col] = S_U[b1 * 2 + b2, k] * L_U[i * 3 + j, l]
                            
                            # V index
                            idx_v = (b1 * 3 + i) * 6 + (b2 * 3 + j)
                            V_6[idx_v, col] = S_V[b1 * 2 + b2, k] * L_V[i * 3 + j, l]
                            
                            # W index
                            # W corresponds to C[k,i]
                            # Block indices for C: b_k (row block), b_i (col block)
                            # In Strassen, C blocks are combined.
                            # S_W row index is b_k * 2 + b_i
                            # L_W row index is k_sub * 3 + i_sub
                            # W_6 row index: (b_k * 3 + k_sub) * 6 + (b_i * 3 + i_sub)
                            # Note: The indices in S_W correspond to C_{b_k b_i}
                            # So b_k is the row block of C, b_i is the col block of C.
                            # In the loop above, we used b1, b2 for A and B.
                            # For W, we need to iterate over all 4 blocks of C.
                            pass
    
    # Re-construct W_6 correctly
    for k in range(7):
        for l in range(7):
            col = k * 7 + l
            for bk in range(2): # C row block
                for bi in range(2): # C col block
                    for i in range(3): # C sub row (k_sub)
                        for j in range(3): # C sub col (i_sub)
                            idx_w = (bk * 3 + i) * 6 + (bi * 3 + j)
                            W_6[idx_w, col] = S_W[bk * 2 + bi, k] * L_W[i * 3 + j, l]
    
    # Extract 5x5
    # U rows 0..4
    U_5 = U_6[0:5, :]
    V_5 = V_6[0:5, :]
    
    # W rows: idx = k*6 + i where k<5, i<5
    w_mask = []
    for k in range(5):
        for i in range(5):
            w_mask.append(k * 6 + i)
    W_5 = W_6[w_mask, :]
    
    return {"U": U_5.tolist(), "V": V_5.tolist(), "W": W_5.tolist()}
# EVOLVE-BLOCK-END
