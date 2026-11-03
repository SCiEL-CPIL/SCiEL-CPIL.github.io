import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Initialize U, V, W as lists of lists, or use numpy for easier accumulation then convert
    # Using lists to match expected output format if strict, but numpy is faster for construction
    # The interface expects {'U': ..., 'V': ..., 'W': ...} with shapes (n*m, rank), etc.
    # I will accumulate in numpy arrays with a generous max rank, then trim.
    
    # Max possible rank is 125. My estimate is 112.
    max_rank = 125
    U = np.zeros((n*m, max_rank), dtype=float)
    V = np.zeros((m*p, max_rank), dtype=float)
    W = np.zeros((p*n, max_rank), dtype=float)
    
    current_rank = 0
    
    def add_block(u_block, v_block, w_block, i_offset, j_offset, k_offset):
        """
        Adds a block decomposition to the global tensors.
        u_block: (n_b, r)
        v_block: (m_b, r)
        w_block: (p_b, r)
        i_offset: offset for i in A and C (row index in A, col index in C)
        j_offset: offset for j in A and B (col index in A, row index in B)
        k_offset: offset for k in B and C (col index in B, row index in C)
        
        Global A index: i*m + j
        Global B index: j*p + k
        Global C index: k*n + i
        """
        nonlocal current_rank
        r = u_block.shape[1]
        
        # Check time
        if time.time() > deadline:
            return False
            
        # Map indices
        # A rows: i_offset .. i_offset + n_b - 1
        # A cols: j_offset .. j_offset + m_b - 1
        # B rows: j_offset .. j_offset + m_b - 1
        # B cols: k_offset .. k_offset + p_b - 1
        # C rows: k_offset .. k_offset + p_b - 1
        # C cols: i_offset .. i_offset + n_b - 1
        
        n_b = u_block.shape[0]
        m_b = v_block.shape[0]
        p_b = w_block.shape[0]
        
        for t in range(r):
            col = current_rank
            if col >= max_rank:
                return False
                
            # Fill U column
            for i in range(n_b):
                for j in range(m_b):
                    u_val = u_block[i, j, t] if u_block.ndim == 3 else u_block[i*m_b+j, t]
                    # u_block is passed as flattened (n_b*m_b, r) or (n_b, m_b, r)?
                    # Let's assume (n_b*m_b, r) for U and V, and (p_b*n_b, r) for W?
                    # No, standard tensor decomposition U is (n*m, r).
                    # So u_block should be (n_b*m_b, r).
                    # v_block (m_b*p_b, r).
                    # w_block (p_b*n_b, r).
                    
            # I will pass pre-flattened blocks.
            pass
        
        # Simpler approach:
        # u_block is (n_b*m_b, r)
        # v_block is (m_b*p_b, r)
        # w_block is (p_b*n_b, r)
        
        # Indices in global U:
        # Row index = i*m + j
        # i in [i_offset, i_offset+n_b), j in [j_offset, j_offset+m_b)
        # This is not a contiguous block in U unless m_b == m?
        # Yes, m_b is a subset of m.
        # We need to scatter.
        
        for t in range(r):
            col = current_rank
            if col >= max_rank:
                return False
                
            # U
            for i in range(n_b):
                for j in range(m_b):
                    global_u_idx = (i_offset + i) * m + (j_offset + j)
                    U[global_u_idx, col] = u_block[i*m_b + j, t]
                    
            # V
            for j in range(m_b):
                for k in range(p_b):
                    global_v_idx = (j_offset + j) * p + (k_offset + k)
                    V[global_v_idx, col] = v_block[j*p_b + k, t]
                    
            # W
            for k in range(p_b):
                for i in range(n_b):
                    global_w_idx = (k_offset + k) * n + (i_offset + i)
                    W[global_w_idx, col] = w_block[k*n_b + i, t]
                    
            current_rank += 1
            
        return True

    # Helper to create Strassen 2x2x2
    def get_strassen_2x2():
        # Returns U, V, W of shape (4, 7), (4, 7), (4, 7)
        # Indices: 0:11, 1:12, 2:21, 3:22
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = -1; W_s[3, 0] = 1
        # Wait, W indices: 0:11, 1:12, 2:21, 3:22
        # C11 = M1 + M4 - M5 + M7
        # C12 = M2 + M3 - M6 + M7
        # C21 = M2 + M5 - M3 - M1
        # C22 = M1 - M2 + M4 + M6
        
        # Let's define the 7 terms M1..M7
        # M1: (A11+A22) * (B11+B22)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1  # C11 += M1
        W_s[1, 0] = 0
        W_s[2, 0] = 0
        W_s[3, 0] = 1  # C22 += M1
        
        # M2: (A11+A12) * B22
        U_s[0, 1] = 1; U_s[1, 1] = 1
        V_s[3, 1] = 1
        W_s[1, 1] = 1  # C12 += M2
        W_s[2, 1] = 1  # C21 += M2
        W_s[3, 1] = -1 # C22 -= M2
        
        # M3: A11 * (B21-B22)
        U_s[0, 2] = 1
        V_s[2, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1  # C12 += M3
        W_s[2, 2] = -1 # C21 -= M3
        
        # M4: A22 * (B11-B12)
        U_s[3, 3] = 1
        V_s[0, 3] = 1; V_s[1, 3] = -1
        W_s[0, 3] = 1  # C11 += M4
        W_s[3, 3] = 1  # C22 += M4
        
        # M5: (A21+A11) * B11
        U_s[0, 4] = 1; U_s[2, 4] = 1
        V_s[0, 4] = 1
        W_s[0, 4] = -1 # C11 -= M5
        W_s[2, 4] = 1  # C21 += M5
        
        # M6: (A21-A11) * (B11+B12)
        U_s[0, 5] = -1; U_s[2, 5] = 1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[1, 5] = -1 # C12 -= M6
        W_s[3, 5] = 1  # C22 += M6
        
        # M7: (A12-A22) * (B21+B22)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1  # C11 += M7
        W_s[1, 6] = 1  # C12 += M7
        
        return U_s, V_s, W_s

    # Helper to create Laderman 3x3x3
    def get_laderman_3x3():
        # Hardcoded Laderman coefficients
        # 15 terms
        U_l = np.zeros((9, 15))
        V_l = np.zeros((9, 15))
        W_l = np.zeros((9, 15))
        
        # Indices 0..8 correspond to 11,12,13, 21,22,23, 31,32,33
        
        # M1 = (A11+A22)(B11+B22)
        U_l[0,0]=1; U_l[4,0]=1
        V_l[0,0]=1; V_l[4,0]=1
        W_l[0,0]=1; W_l[4,0]=1; W_l[1,0]=-1; W_l[3,0]=-1
        
        # M2 = (A11+A12)B22
        U_l[0,1]=1; U_l[1,1]=1
        V_l[4,1]=1
        W_l[1,1]=1; W_l[3,1]=1; W_l[4,1]=-1
        
        # M3 = A11(B21-B22)
        U_l[0,2]=1
        V_l[3,2]=1; V_l[4,2]=-1
        W_l[1,2]=1; W_l[3,2]=-1
        
        # M4 = A22(B11-B12)
        U_l[4,3]=1
        V_l[0,3]=1; V_l[1,3]=-1
        W_l[0,3]=1; W_l[4,3]=1
        
        # M5 = (A21+A11)B11
        U_l[0,4]=1; U_l[3,4]=1
        V_l[0,4]=1
        W_l[0,4]=-1; W_l[3,4]=1
        
        # M6 = (A21-A11)(B11+B12)
        U_l[0,5]=-1; U_l[3,5]=1
        V_l[0,5]=1; V_l[1,5]=1
        W_l[1,5]=-1; W_l[4,5]=1
        
        # M7 = (A12-A22)(B21+B22)
        U_l[1,6]=1; U_l[4,6]=-1
        V_l[3,6]=1; V_l[4,6]=1
        W_l[0,6]=1; W_l[1,6]=1
        
        # M8 = (A13+A33)(B13+B33)
        U_l[2,7]=1; U_l[8,7]=1
        V_l[2,7]=1; V_l[8,7]=1
        W_l[2,7]=1; W_l[8,7]=1; W_l[5,7]=-1; W_l[6,7]=-1
        
        # M9 = (A13+A12)B33
        U_l[1,8]=1; U_l[2,8]=1
        V_l[8,8]=1
        W_l[2,8]=1; W_l[5,8]=1; W_l[8,8]=-1
        
        # M10 = A13(B31-B33)
        U_l[2,9]=1
        V_l[6,9]=1; V_l[8,9]=-1
        W_l[2,9]=1; W_l[6,9]=-1
        
        # M11 = A33(B11-B13)
        U_l[8,10]=1
        V_l[0,10]=1; V_l[2,10]=-1
        W_l[2,10]=1; W_l[8,10]=1
        
        # M12 = (A31+A11)B13
        U_l[0,11]=1; U_l[6,11]=1
        V_l[2,11]=1
        W_l[2,11]=-1; W_l[6,11]=1
        
        # M13 = (A31-A11)(B11+B13)
        U_l[0,12]=-1; U_l[6,12]=1
        V_l[0,12]=1; V_l[2,12]=1
        W_l[2,12]=-1; W_l[8,12]=1
        
        # M14 = (A13-A33)(B31+B33)
        U_l[2,13]=1; U_l[8,13]=-1
        V_l[6,13]=1; V_l[8,13]=1
        W_l[2,13]=1; W_l[5,13]=1
        
        # M15 = (A11+A33)(B11+B33)
        U_l[0,14]=1; U_l[8,14]=1
        V_l[0,14]=1; V_l[8,14]=1
        W_l[0,14]=1; W_l[8,14]=1; W_l[2,14]=-1; W_l[6,14]=-1
        
        return U_l, V_l, W_l

    # Helper to create Schoolbook for nxm x mxp
    def get_schoolbook(n_b, m_b, p_b):
        r = n_b * m_b * p_b
        U_s = np.zeros((n_b*m_b, r))
        V_s = np.zeros((m_b*p_b, r))
        W_s = np.zeros((p_b*n_b, r))
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_s[i*m_b+j, col] = 1
                    V_s[j*p_b+k, col] = 1
                    W_s[k*n_b+i, col] = 1
                    col += 1
        return U_s, V_s, W_s

    # 1. A22 B22 -> C22 (2x2x2)
    U1, V1, W1 = get_strassen_2x2()
    if not add_block(U1, V1, W1, 0, 0, 0): return None

    # 2. A33 B33 -> C33 (3x3x3)
    U2, V2, W2 = get_laderman_3x3()
    if not add_block(U2, V2, W2, 2, 2, 2): return None

    # 3. A23 B32 -> C22 (2x3 x 3x2 -> 2x2)
    # A23: rows 0-1, cols 2-4. B32: rows 2-4, cols 0-1.
    # i_offset=0, j_offset=2, k_offset=0
    U3, V3, W3 = get_schoolbook(2, 3, 2)
    if not add_block(U3, V3, W3, 0, 2, 0): return None

    # 4. A22 B23 -> C23 (2x2 x 2x3 -> 2x3)
    # A22: rows 0-1, cols 0-1. B23: rows 0-1, cols 2-4.
    # i_offset=0, j_offset=0, k_offset=2
    U4, V4, W4 = get_schoolbook(2, 2, 3)
    if not add_block(U4, V4, W4, 0, 0, 2): return None

    # 5. A32 B22 -> C32 (3x2 x 2x2 -> 3x2)
    # A32: rows 2-4, cols 0-1. B22: rows 0-1, cols 0-1.
    # i_offset=2, j_offset=0, k_offset=0
    U5, V5, W5 = get_schoolbook(3, 2, 2)
    if not add_block(U5, V5, W5, 2, 0, 0): return None

    # 6. A23 B33 -> C23 (2x3 x 3x3 -> 2x3)
    # A23: rows 0-1, cols 2-4. B33: rows 2-4, cols 2-4.
    # i_offset=0, j_offset=2, k_offset=2
    U6, V6, W6 = get_schoolbook(2, 3, 3)
    if not add_block(U6, V6, W6, 0, 2, 2): return None

    # 7. A33 B32 -> C32 (3x3 x 3x2 -> 3x2)
    # A33: rows 2-4, cols 2-4. B32: rows 2-4, cols 0-1.
    # i_offset=2, j_offset=2, k_offset=0
    U7, V7, W7 = get_schoolbook(3, 3, 2)
    if not add_block(U7, V7, W7, 2, 2, 0): return None

    # 8. A32 B23 -> C33 (3x2 x 2x3 -> 3x3)
    # A32: rows 2-4, cols 0-1. B23: rows 0-1, cols 2-4.
    # i_offset=2, j_offset=0, k_offset=2
    U8, V8, W8 = get_schoolbook(3, 2, 3)
    if not add_block(U8, V8, W8, 2, 0, 2): return None

    # Trim and convert to lists
    U_final = U[:, :current_rank]
    V_final = V[:, :current_rank]
    W_final = W[:, :current_rank]
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
