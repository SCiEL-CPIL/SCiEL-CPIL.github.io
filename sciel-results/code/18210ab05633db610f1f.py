import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    # n=3, m=3, p=7
    
    # We use a block decomposition:
    # A is 3x3, split into:
    # A22 (2x2), A21 (2x1), A12 (1x2), A11 (1x1)
    # B is 3x7, split into:
    # B27 (2x7), B17 (1x7)
    
    # C_top (2x7) = A22 * B27 + A21 * B17
    # C_bot (1x7) = A12 * B27 + A11 * B17
    
    # We construct the tensor decomposition for each part and concatenate.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a block decomposition
    def add_block(n_block, m_block, p_block, rank_block, U_block, V_block, W_block, 
                  i_offset, j_offset, k_offset):
        # U_block is (n_block*m_block, rank_block)
        # V_block is (m_block*p_block, rank_block)
        # W_block is (p_block*n_block, rank_block)
        
        # We need to map indices to the global 3x3x7 tensor.
        # Global U index: i*m + j
        # Global V index: j*p + k
        # Global W index: k*n + i
        
        for r in range(rank_block):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            
            for i_local in range(n_block):
                for j_local in range(m_block):
                    u_global = (i_offset + i_local) * m + (j_offset + j_local)
                    u_vec[u_global] = U_block[i_local * m_block + j_local][r]
            
            for j_local in range(m_block):
                for k_local in range(p_block):
                    v_global = (j_offset + j_local) * p + (k_offset + k_local)
                    v_vec[v_global] = V_block[j_local * p_block + k_local][r]
            
            for k_local in range(p_block):
                for i_local in range(n_block):
                    w_global = (k_offset + k_local) * n + (i_offset + i_local)
                    w_vec[w_global] = W_block[k_local * n_block + i_local][r]
            
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
    
    # 1. A22 (2x2) * B27 (2x7)
    # Split B27 into B22_1 (2x2), B22_2 (2x2), B21 (2x1)
    # B27 columns: 0,1,2,3,4,5,6
    # B22_1: cols 0,1
    # B22_2: cols 2,3
    # B21: cols 4,5,6 (wait, 2x1 is 2 rows, 1 col. But we have 3 cols left. 
    # We can split B27 into 2x2, 2x2, 2x2, 2x1? 
    # 7 = 2+2+2+1.
    # So 3 blocks of 2x2 and 1 block of 2x1.
    
    # Block 1: A22 * B22_1 (2x2 by 2x2)
    # Use Strassen (rank 7)
    U_222 = [[0]*7 for _ in range(4)]
    V_222 = [[0]*7 for _ in range(4)]
    W_222 = [[0]*7 for _ in range(4)]
    # Strassen coefficients for 2x2x2
    # c1 = a11(b11+b22) -> U: a11, V: b11+b22, W: c11
    # This is complex to hardcode correctly.
    # Let's use schoolbook for 2x2x2 (rank 4) to be safe?
    # Rank 4 is better than 7? No, 7 is for 2x2x2.
    # Schoolbook for 2x2x2 is rank 4.
    # Strassen is rank 7.
    # So schoolbook is better for 2x2x2!
    
    # Wait, Strassen reduces 8 to 7.
    # Schoolbook is 4.
    # So for 2x2x2, schoolbook is rank 4.
    
    # So for A22 * B22_1 (2x2 by 2x2), rank 4.
    # For A22 * B22_2 (2x2 by 2x2), rank 4.
    # For A22 * B22_3 (2x2 by 2x2), rank 4.
    # For A22 * B21 (2x2 by 2x1), rank 4.
    
    # Total for A22 * B27: 4+4+4+4 = 16.
    
    # 2. A21 (2x1) * B17 (1x7)
    # Rank 2*7 = 14.
    
    # 3. A12 (1x2) * B27 (2x7)
    # Rank 1*7 = 7? No, 2*7=14.
    
    # 4. A11 (1x1) * B17 (1x7)
    # Rank 1*7 = 7.
    
    # Total rank: 16 + 14 + 14 + 7 = 51.
    
    # This is better than 63.
    
    # Let's implement schoolbook for each block.
    
    def schoolbook(n_b, m_b, p_b):
        rank = n_b * m_b * p_b
        U = [[0]*rank for _ in range(n_b * m_b)]
        V = [[0]*rank for _ in range(m_b * p_b)]
        W = [[0]*rank for _ in range(p_b * n_b)]
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[i*m_b+j][col] = 1
                    V[j*p_b+k][col] = 1
                    W[k*n_b+i][col] = 1
                    col += 1
        return U, V, W
    
    # Block 1: A22 (2x2) * B22_1 (2x2)
    U1, V1, W1 = schoolbook(2, 2, 2)
    add_block(2, 2, 2, 4, U1, V1, W1, 0, 0, 0)
    
    # Block 2: A22 (2x2) * B22_2 (2x2)
    U2, V2, W2 = schoolbook(2, 2, 2)
    add_block(2, 2, 2, 4, U2, V2, W2, 0, 0, 2)
    
    # Block 3: A22 (2x2) * B22_3 (2x2)
    U3, V3, W3 = schoolbook(2, 2, 2)
    add_block(2, 2, 2, 4, U3, V3, W3, 0, 0, 4)
    
    # Block 4: A22 (2x2) * B21 (2x1)
    U4, V4, W4 = schoolbook(2, 2, 1)
    add_block(2, 2, 1, 4, U4, V4, W4, 0, 0, 6)
    
    # Block 5: A21 (2x1) * B17 (1x7)
    U5, V5, W5 = schoolbook(2, 1, 7)
    add_block(2, 1, 7, 14, U5, V5, W5, 0, 2, 0)
    
    # Block 6: A12 (1x2) * B27 (2x7)
    U6, V6, W6 = schoolbook(1, 2, 7)
    add_block(1, 2, 7, 14, U6, V6, W6, 2, 0, 0)
    
    # Block 7: A11 (1x1) * B17 (1x7)
    U7, V7, W7 = schoolbook(1, 1, 7)
    add_block(1, 1, 7, 7, U7, V7, W7, 2, 2, 0)
    
    rank = len(U_list)
    
    # Verify
    U = np.array(U_list)
    V = np.array(V_list)
    W = np.array(W_list)
    
    # Check time
    if time.time() > deadline - 10:
        # Fallback to schoolbook if running out of time
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
        
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
