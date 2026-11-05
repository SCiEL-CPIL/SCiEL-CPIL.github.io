import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Aims for rank 54 using rank-13 decomposition for 2x4x2 blocks.
    """
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_indices, v_indices, w_indices, U_blk, V_blk, W_blk):
        rank = U_blk.shape[1]
        for r in range(rank):
            u_vec = np.zeros(n * m)
            v_vec = np.zeros(m * p)
            w_vec = np.zeros(p * n)
            
            for i, idx in enumerate(u_indices):
                u_vec[idx] = U_blk[i, r]
            for i, idx in enumerate(v_indices):
                v_vec[idx] = V_blk[i, r]
            for i, idx in enumerate(w_indices):
                w_vec[idx] = W_blk[i, r]
                
            U_parts.append(u_vec)
            V_parts.append(v_vec)
            W_parts.append(w_vec)

    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # Rank-13 decomposition for 2x4x2 (Middle Product 4)
    # This is a placeholder. I will use rank-14 (two Strassen) if I can't find 13.
    # For now, I will use the rank-14 logic but structure it as a single block.
    def middle_product_2x4x2_rank14():
        # This is just two Strassen blocks combined.
        # U is 4x14, V is 4x14, W is 4x14? No.
        # U is 4 (2x2 A) x 14.
        # V is 4 (4x2 B) x 14.
        # W is 4 (2x2 C) x 14.
        U = np.zeros((4, 14))
        V = np.zeros((4, 14))
        W = np.zeros((4, 14))
        
        # Block 1: A[0:2, 0:2] * B[0:2, :] (2x2x2)
        U_b, V_b, W_b = strassen_2x2x2()
        # Map indices
        # U_b is 4x7. Indices 0,1,2,3 correspond to A[0,0], A[0,1], A[1,0], A[1,1].
        # In our 2x4 A, these are A[0,0], A[0,1], A[1,0], A[1,1].
        # So U[0:4, 0:7] = U_b.
        U[:, 0:7] = U_b
        
        # V_b is 4x7. Indices 0,1,2,3 correspond to B[0,0], B[1,0], B[0,1], B[1,1].
        # In our 4x2 B, these are B[0,0], B[1,0], B[0,1], B[1,1].
        # Wait, B is 4x2. Rows 0,1 are the first 2 rows.
        # So V[0:4, 0:7] = V_b.
        V[:, 0:7] = V_b
        
        # W_b is 4x7. Indices 0,1,2,3 correspond to C[0,0], C[0,1], C[1,0], C[1,1].
        # In our 2x2 C, these are C[0,0], C[0,1], C[1,0], C[1,1].
        # So W[0:4, 0:7] = W_b.
        W[:, 0:7] = W_b
        
        # Block 2: A[0:2, 2:4] * B[2:4, :] (2x2x2)
        U_b2, V_b2, W_b2 = strassen_2x2x2()
        # U_b2 indices 0,1,2,3 correspond to A[0,2], A[0,3], A[1,2], A[1,3].
        # In our 2x4 A, these are A[0,2], A[0,3], A[1,2], A[1,3].
        # So U[0:4, 7:14] = U_b2.
        U[:, 7:14] = U_b2
        
        # V_b2 indices 0,1,2,3 correspond to B[2,0], B[3,0], B[2,1], B[3,1].
        # In our 4x2 B, these are B[2,0], B[3,0], B[2,1], B[3,1].
        # So V[0:4, 7:14] = V_b2.
        V[:, 7:14] = V_b2
        
        # W_b2 indices 0,1,2,3 correspond to C[0,0], C[0,1], C[1,0], C[1,1].
        # In our 2x2 C, these are C[0,0], C[0,1], C[1,0], C[1,1].
        # So W[0:4, 7:14] = W_b2.
        W[:, 7:14] = W_b2
        
        return U, V, W

    # Indices
    # U: 3x4 -> 12
    # V: 4x5 -> 20
    # W: 5x3 -> 15
    
    # Part 1: A[0:2, :] * B
    # A_top = A[0:2, :] (2x4)
    # B is 4x5.
    # We partition B into columns 0:2, 2:4, 4:5.
    # Block 1: A_top * B[:, 0:2] (2x4x2)
    U_b, V_b, W_b = middle_product_2x4x2_rank14()
    add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4], U_b, V_b, W_b)
    
    # Block 2: A_top * B[:, 2:4] (2x4x2)
    U_b, V_b, W_b = middle_product_2x4x2_rank14()
    add_block([0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10], U_b, V_b, W_b)
    
    # Block 3: A_top * B[:, 4:5] (2x4x1)
    # This is schoolbook 2x4x1. Rank 8.
    # U is 4x8, V is 4x8, W is 2x8.
    U_b = np.zeros((4, 8))
    V_b = np.zeros((4, 8))
    W_b = np.zeros((2, 8))
    for i in range(4):
        for j in range(2):
            idx = i * 2 + j
            U_b[i, idx] = 1
            V_b[i, idx] = 1
            W_b[j, idx] = 1
    add_block([0, 1, 4, 5], [4, 9, 14, 19], [12, 13], U_b, V_b, W_b)
    
    # Part 2: A[2, :] * B
    # A_bot = A[2, :] (1x4)
    # Block 4: A_bot * B[:, 0:2] (1x4x2)
    # Schoolbook 1x4x2. Rank 8.
    U_b = np.zeros((4, 8))
    V_b = np.zeros((4, 8))
    W_b = np.zeros((2, 8))
    for i in range(4):
        for j in range(2):
            idx = i * 2 + j
            U_b[i, idx] = 1
            V_b[i, idx] = 1
            W_b[j, idx] = 1
    add_block([8, 9, 10, 11], [0, 1, 5, 6], [2, 5], U_b, V_b, W_b)
    
    # Block 5: A_bot * B[:, 2:4] (1x4x2)
    U_b = np.zeros((4, 8))
    V_b = np.zeros((4, 8))
    W_b = np.zeros((2, 8))
    for i in range(4):
        for j in range(2):
            idx = i * 2 + j
            U_b[i, idx] = 1
            V_b[i, idx] = 1
            W_b[j, idx] = 1
    add_block([8, 9, 10, 11], [2, 3, 7, 8], [8, 11], U_b, V_b, W_b)
    
    # Block 6: A_bot * B[:, 4:5] (1x4x1)
    # Schoolbook 1x4x1. Rank 4.
    U_b = np.zeros((4, 4))
    V_b = np.zeros((4, 4))
    W_b = np.zeros((1, 4))
    for i in range(4):
        U_b[i, i] = 1
        V_b[i, i] = 1
        W_b[0, i] = 1
    add_block([8, 9, 10, 11], [4, 9, 14, 19], [14], U_b, V_b, W_b)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
