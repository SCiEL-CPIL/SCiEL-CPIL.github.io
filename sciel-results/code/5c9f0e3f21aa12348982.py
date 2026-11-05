import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    _start_time = time.time()
    _deadline = _start_time + 120
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 32
    pn = p * n  # 24
    
    terms = []
    
    def time_check():
        return time.time() > _deadline
        
    def add_strassen_block(i0, j0, k0):
        """
        Adds the 7 rank-1 terms for the Strassen decomposition of
        A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        if time_check():
            return
            
        # A indices (row, col) -> index
        a00_u = i0 * m + j0
        a01_u = i0 * m + (j0 + 1)
        a10_u = (i0 + 1) * m + j0
        a11_u = (i0 + 1) * m + (j0 + 1)
        
        # B indices (row, col) -> index
        b00_v = j0 * p + k0
        b01_v = j0 * p + (k0 + 1)
        b10_v = (j0 + 1) * p + k0
        b11_v = (j0 + 1) * p + (k0 + 1)
        
        # C indices (row, col) -> index (W[k*n + i])
        c00_w = k0 * n + i0
        c01_w = (k0 + 1) * n + i0
        c10_w = k0 * n + (i0 + 1)
        c11_w = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a00 + a11) * (b00 + b11)
        u1 = [0.0] * nm; u1[a00_u] = 1.0; u1[a11_u] = 1.0
        v1 = [0.0] * mp; v1[b00_v] = 1.0; v1[b11_v] = 1.0
        w1 = [0.0] * pn; w1[c00_w] = 1.0; w1[c11_w] = 1.0
        terms.append((u1, v1, w1))
        
        # M2 = (a10 + a11) * b00
        u2 = [0.0] * nm; u2[a10_u] = 1.0; u2[a11_u] = 1.0
        v2 = [0.0] * mp; v2[b00_v] = 1.0
        w2 = [0.0] * pn; w2[c10_w] = 1.0; w2[c11_w] = -1.0
        terms.append((u2, v2, w2))
        
        # M3 = a00 * (b01 - b11)
        u3 = [0.0] * nm; u3[a00_u] = 1.0
        v3 = [0.0] * mp; v3[b01_v] = 1.0; v3[b11_v] = -1.0
        w3 = [0.0] * pn; w3[c01_w] = 1.0; w3[c11_w] = 1.0
        terms.append((u3, v3, w3))
        
        # M4 = a11 * (b10 - b00)
        u4 = [0.0] * nm; u4[a11_u] = 1.0
        v4 = [0.0] * mp; v4[b10_v] = 1.0; v4[b00_v] = -1.0
        w4 = [0.0] * pn; w4[c00_w] = 1.0; w4[c10_w] = 1.0
        terms.append((u4, v4, w4))
        
        # M5 = (a00 + a01) * b11
        u5 = [0.0] * nm; u5[a00_u] = 1.0; u5[a01_u] = 1.0
        v5 = [0.0] * mp; v5[b11_v] = 1.0
        w5 = [0.0] * pn; w5[c00_w] = -1.0; w5[c01_w] = 1.0
        terms.append((u5, v5, w5))
        
        # M6 = (a10 - a00) * (b00 + b01)
        u6 = [0.0] * nm; u6[a10_u] = 1.0; u6[a00_u] = -1.0
        v6 = [0.0] * mp; v6[b00_v] = 1.0; v6[b01_v] = 1.0
        w6 = [0.0] * pn; w6[c11_w] = 1.0
        terms.append((u6, v6, w6))
        
        # M7 = (a01 - a11) * (b10 + b11)
        u7 = [0.0] * nm; u7[a01_u] = 1.0; u7[a11_u] = -1.0
        v7 = [0.0] * mp; v7[b10_v] = 1.0; v7[b11_v] = 1.0
        w7 = [0.0] * pn; w7[c00_w] = 1.0
        terms.append((u7, v7, w7))

    def add_schoolbook_1x2_2x2(i_row, j0, k0):
        """
        Adds the 4 rank-1 terms for the schoolbook decomposition of
        A[i_row, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        A is 1x2, B is 2x2, C is 1x2 (contributes to row i_row of C)
        """
        if time_check():
            return
            
        for j in range(2):
            for k in range(2):
                u_idx = i_row * m + (j0 + j)
                v_idx = (j0 + j) * p + (k0 + k)
                w_idx = (k0 + k) * n + i_row
                
                u_vec = [0.0] * nm; u_vec[u_idx] = 1.0
                v_vec = [0.0] * mp; v_vec[v_idx] = 1.0
                w_vec = [0.0] * pn; w_vec[w_idx] = 1.0
                terms.append((u_vec, v_vec, w_vec))

    # B is 4x8. Split into B1 (4x4) and B2 (4x4).
    # For each B block, we process A (3x4).
    # A is split into A_top (2x4) and A_bot (1x4).
    
    # Process B1 (columns 0-3 of B, i.e., k=0..3)
    # A_top (rows 0-1 of A) * B1
    # B1 is 4x4. Split B1 into 2x2 blocks:
    # B1_00 (j=0,1, k=0,1), B1_01 (j=0,1, k=2,3)
    # B1_10 (j=2,3, k=0,1), B1_11 (j=2,3, k=2,3)
    # A_top is 2x4. Split into A_top_0 (cols 0,1) and A_top_1 (cols 2,3).
    #
    # C_top_left (cols 0,1 of C) comes from A_top_0 * B1_00 + A_top_1 * B1_10
    # C_top_right (cols 2,3 of C) comes from A_top_0 * B1_01 + A_top_1 * B1_11
    
    # Strassen blocks for A_top * B1
    # Block (i0=0, j0=0, k0=0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0)
    # Block (i0=0, j0=0, k0=2): A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 2)
    # Block (i0=0, j0=2, k0=0): A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 2, 0)
    # Block (i0=0, j0=2, k0=2): A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 2)
    
    # Schoolbook blocks for A_bot * B1
    # A_bot is row 2 of A.
    # A_bot_0 (cols 0,1) * B1_00 (j=0,1, k=0,1) -> C[2, 0:2]
    add_schoolbook_1x2_2x2(2, 0, 0)
    # A_bot_0 (cols 0,1) * B1_01 (j=0,1, k=2,3) -> C[2, 2:4]
    add_schoolbook_1x2_2x2(2, 0, 2)
    # A_bot_1 (cols 2,3) * B1_10 (j=2,3, k=0,1) -> C[2, 0:2]
    add_schoolbook_1x2_2x2(2, 2, 0)
    # A_bot_1 (cols 2,3) * B1_11 (j=2,3, k=2,3) -> C[2, 2:4]
    add_schoolbook_1x2_2x2(2, 2, 2)
    
    # Process B2 (columns 4-7 of B, i.e., k=4..7)
    # A_top * B2
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 0, 6)
    add_strassen_block(0, 2, 4)
    add_strassen_block(0, 2, 6)
    
    # A_bot * B2
    add_schoolbook_1x2_2x2(2, 0, 4)
    add_schoolbook_1x2_2x2(2, 0, 6)
    add_schoolbook_1x2_2x2(2, 2, 4)
    add_schoolbook_1x2_2x2(2, 2, 6)

    rank = len(terms)
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for r in range(rank):
        u_vec, v_vec, w_vec = terms[r]
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
