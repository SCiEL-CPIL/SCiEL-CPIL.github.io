import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We build the decomposition block by block.
    # Each block returns U_block, V_block, W_block.
    # We concatenate them.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(n_b, m_b, p_b, i_off, j_off, k_off, use_strassen=False):
        """
        Adds a block of size n_b x m_b x p_b at offsets i_off, j_off, k_off.
        If use_strassen is True and n_b=m_b=p_b=2, uses Strassen.
        Otherwise uses schoolbook.
        """
        # Local dimensions
        # Global indices
        # a = (i_off + i') * m + (j_off + j')
        # b = (j_off + j') * p + (k_off + k')
        # c = (k_off + k') * n + (i_off + i')
        
        if use_strassen and n_b == 2 and m_b == 2 and p_b == 2:
            # Strassen 2x2x2
            # 7 multiplications
            # U_block shape (n*m, 7) -- actually we need to place into global U
            # It's easier to create full global size matrices with zeros and fill in
            
            # Strassen coefficients
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # We will create arrays for the 7 terms
            # U, V, W for this block will be added to the global accumulators
            
            # Initialize local U, V, W for this block
            # But we need to map to global indices
            
            # Let's define the 7 terms' contributions to U, V, W
            # Term 1
            u1 = {
                (i_off+0, j_off+0): 1,
                (i_off+1, j_off+1): 1
            }
            v1 = {
                (j_off+0, k_off+0): 1,
                (j_off+1, k_off+1): 1
            }
            w1 = {
                (k_off+0, i_off+0): 1,
                (k_off+1, i_off+1): 1
            }
            
            # Term 2
            u2 = {
                (i_off+1, j_off+0): 1,
                (i_off+1, j_off+1): 1
            }
            v2 = {
                (j_off+0, k_off+0): 1
            }
            w2 = {
                (k_off+0, i_off+1): 1  # C10
            }
            
            # Term 3
            u3 = {
                (i_off+0, j_off+0): 1
            }
            v3 = {
                (j_off+1, k_off+0): 1,
                (j_off+1, k_off+1): -1
            }
            w3 = {
                (k_off+0, i_off+0): 1,  # C00
                (k_off+1, i_off+0): 1   # C01
            }
            
            # Term 4
            u4 = {
                (i_off+1, j_off+1): 1
            }
            v4 = {
                (j_off+0, k_off+1): 1,
                (j_off+0, k_off+0): -1
            }
            w4 = {
                (k_off+0, i_off+0): 1,  # C00
                (k_off+1, i_off+1): 1   # C11
            }
            
            # Term 5
            u5 = {
                (i_off+0, j_off+0): 1,
                (i_off+0, j_off+1): 1
            }
            v5 = {
                (j_off+1, k_off+1): 1
            }
            w5 = {
                (k_off+0, i_off+0): -1, # C00
                (k_off+1, i_off+0): 1   # C01
            }
            
            # Term 6
            u6 = {
                (i_off+1, j_off+0): 1,
                (i_off+0, j_off+0): -1
            }
            v6 = {
                (j_off+0, k_off+0): 1,
                (j_off+0, k_off+1): 1
            }
            w6 = {
                (k_off+1, i_off+1): 1   # C11
            }
            
            # Term 7
            u7 = {
                (i_off+0, j_off+1): 1,
                (i_off+1, j_off+1): -1
            }
            v7 = {
                (j_off+1, k_off+0): 1,
                (j_off+1, k_off+1): 1
            }
            w7 = {
                (k_off+0, i_off+0): 1,  # C00
                (k_off+1, i_off+1): 1   # C11
            }
            
            terms = [
                (u1, v1, w1),
                (u2, v2, w2),
                (u3, v3, w3),
                (u4, v4, w4),
                (u5, v5, w5),
                (u6, v6, w6),
                (u7, v7, w7)
            ]
            
            for u_t, v_t, w_t in terms:
                U_col = np.zeros(n * m)
                V_col = np.zeros(m * p)
                W_col = np.zeros(p * n)
                
                for (i_loc, j_loc), val in u_t.items():
                    # i_loc, j_loc are global i, j
                    a = i_loc * m + j_loc
                    U_col[a] = val
                
                for (j_loc, k_loc), val in v_t.items():
                    b = j_loc * p + k_loc
                    V_col[b] = val
                
                for (k_loc, i_loc), val in w_t.items():
                    c = k_loc * n + i_loc
                    W_col[c] = val
                
                U_parts.append(U_col)
                V_parts.append(V_col)
                W_parts.append(W_col)
                
        else:
            # Schoolbook
            for i_loc in range(n_b):
                for j_loc in range(m_b):
                    for k_loc in range(p_b):
                        i_g = i_off + i_loc
                        j_g = j_off + j_loc
                        k_g = k_off + k_loc
                        
                        a = i_g * m + j_g
                        b = j_g * p + k_g
                        c = k_g * n + i_g
                        
                        U_col = np.zeros(n * m)
                        V_col = np.zeros(m * p)
                        W_col = np.zeros(p * n)
                        
                        U_col[a] = 1
                        V_col[b] = 1
                        W_col[c] = 1
                        
                        U_parts.append(U_col)
                        V_parts.append(V_col)
                        W_parts.append(W_col)

    # Block 1: n[0:2], m[0:4], p[0:6]
    # Split m into [0:2], [2:4]
    # Split p into [0:2], [2:4], [4:6]
    # 6 blocks of 2x2x2
    for i in range(0, 2): # m offset 0, 2
        for k in range(0, 6, 2): # p offset 0, 2, 4
            add_block(2, 2, 2, 0, i, k, use_strassen=True)

    # Block 2: n[0:2], m[4:5], p[0:6]
    # 1x2x6 -> 3 blocks of 1x2x2? No, 2x1x2 blocks.
    # n is 2, m is 1, p is 2.
    for k in range(0, 6, 2):
        add_block(2, 1, 2, 0, 4, k, use_strassen=False)

    # Block 3: n[0:2], m[0:4], p[6:7]
    # 2x4x1 -> 2 blocks of 2x2x1
    for i in range(0, 4, 2):
        add_block(2, 2, 1, 0, i, 6, use_strassen=False)

    # Block 4: n[0:2], m[4:5], p[6:7]
    # 2x1x1
    add_block(2, 1, 1, 0, 4, 6, use_strassen=False)

    # Block 5: n[2:3], m[0:4], p[0:6]
    # 1x4x6 -> 6 blocks of 1x2x2
    for i in range(0, 4, 2):
        for k in range(0, 6, 2):
            add_block(1, 2, 2, 2, i, k, use_strassen=False)

    # Block 6: n[2:3], m[4:5], p[0:6]
    # 1x1x6 -> 6 blocks of 1x1x1
    for k in range(0, 6):
        add_block(1, 1, 1, 2, 4, k, use_strassen=False)

    # Block 7: n[2:3], m[0:4], p[6:7]
    # 1x4x1 -> 4 blocks of 1x1x1
    for i in range(0, 4):
        add_block(1, 1, 1, 2, i, 6, use_strassen=False)

    # Block 8: n[2:3], m[4:5], p[6:7]
    # 1x1x1
    add_block(1, 1, 1, 2, 4, 6, use_strassen=False)

    U = np.column_stack(U_parts)
    V = np.column_stack(V_parts)
    W = np.column_stack(W_parts)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
