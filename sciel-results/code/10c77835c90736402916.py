import numpy as np

def construct():
    n = 4
    rank = 49
    
    # Initialize U, V, W
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Helper to get flattened index for A[i, k]
    def a_idx(i, k):
        return i * n + k
        
    # Helper to get flattened index for B[k, j]
    def b_idx(k, j):
        return k * n + j
        
    # Helper to get flattened index for C[i, j]
    def c_idx(i, j):
        return i * n + j

    # Define the 7 Strassen steps for 4x4
    # Each step r has X_r and Y_r
    # X_r and Y_r are 2x2 matrices of blocks
    # We define the coefficients for each element x_ij and y_ij
    # as lists of (index, coeff)
    
    # Block definitions for A
    # A00: (0,0), (0,1), (1,0), (1,1) -> idx 0,1,4,5
    # A01: (0,2), (0,3), (1,2), (1,3) -> idx 2,3,6,7
    # A10: (2,0), (2,1), (3,0), (3,1) -> idx 8,9,12,13
    # A11: (2,2), (2,3), (3,2), (3,3) -> idx 10,11,14,15
    
    # We represent x_ij as a dict {flat_idx: coeff}
    
    def block_A(b_row, b_col):
        # b_row, b_col in {0,1}
        # Returns dict for the 2x2 block
        base_i = b_row * 2
        base_k = b_col * 2
        return {
            (base_i, base_k): 1,
            (base_i, base_k+1): 1,
            (base_i+1, base_k): 1,
            (base_i+1, base_k+1): 1
        }
        
    def block_B(b_row, b_col):
        base_k = b_row * 2
        base_j = b_col * 2
        return {
            (base_k, base_j): 1,
            (base_k, base_j+1): 1,
            (base_k+1, base_j): 1,
            (base_k+1, base_j+1): 1
        }
        
    def add_blocks(d1, d2, sign=1):
        res = {}
        for k, v in d1.items():
            res[k] = res.get(k, 0) + v
        for k, v in d2.items():
            res[k] = res.get(k, 0) + sign * v
        return res
        
    def sub_blocks(d1, d2):
        return add_blocks(d1, d2, -1)

    # Define X_r and Y_r for r=0..6
    # r=0: M1 = (A00+A11)(B00+B11)
    X0 = [
        add_blocks(block_A(0,0), block_A(1,1)), # x00
        add_blocks(block_A(0,0), block_A(1,1)), # x01
        add_blocks(block_A(0,0), block_A(1,1)), # x10
        add_blocks(block_A(0,0), block_A(1,1))  # x11
    ]
    # Wait, A00+A11 means element-wise addition.
    # x00 = A00[0,0] + A11[0,0]
    # My block_A returns the 4 elements.
    # So X0[0] should be the sum of the top-left elements of A00 and A11.
    # My block_A(0,0) returns {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    # This is wrong. I need to extract specific elements.
    
    # Let's refine.
    # x_ij is a scalar. It is a linear combination of A elements.
    # For X0 = A00 + A11:
    # x00 = A[0,0] + A[2,2]
    # x01 = A[0,1] + A[2,3]
    # x10 = A[1,0] + A[3,2]
    # x11 = A[1,1] + A[3,3]
    
    def get_x(r, i, j):
        # r: step index
        # i, j: 0 or 1 (position in 2x2 block)
        # Returns dict {a_flat_idx: coeff}
        
        if r == 0: # A00 + A11
            return {a_idx(i, j): 1, a_idx(i+2, j+2): 1}
        elif r == 1: # A10 + A11
            return {a_idx(i+2, j): 1, a_idx(i+2, j+2): 1}
        elif r == 2: # A00
            return {a_idx(i, j): 1}
        elif r == 3: # A11
            return {a_idx(i+2, j+2): 1}
        elif r == 4: # A00 + A01
            return {a_idx(i, j): 1, a_idx(i, j+2): 1}
        elif r == 5: # A10 - A00
            return {a_idx(i+2, j): 1, a_idx(i, j): -1}
        elif r == 6: # A01 - A11
            return {a_idx(i, j+2): 1, a_idx(i+2, j+2): -1}
            
    def get_y(r, i, j):
        # i, j: 0 or 1 (position in 2x2 block of B)
        # Note: B is indexed by (k, j).
        # In the 2x2 block structure, the "row" of the block corresponds to k/2, "col" to j/2.
        # So y_ij corresponds to B element at (2*i + i_offset, 2*j + j_offset)?
        # No, the 2x2 matrix Y_r has elements y_00, y_01, y_10, y_11.
        # y_00 corresponds to the top-left element of the B block.
        
        if r == 0: # B00 + B11
            return {b_idx(i, j): 1, b_idx(i+2, j+2): 1}
        elif r == 1: # B00
            return {b_idx(i, j): 1}
        elif r == 2: # B01 - B11
            return {b_idx(i, j+2): 1, b_idx(i+2, j+2): -1}
        elif r == 3: # B10 - B00
            return {b_idx(i+2, j): 1, b_idx(i, j): -1}
        elif r == 4: # B11
            return {b_idx(i+2, j+2): 1}
        elif r == 5: # B00 + B01
            return {b_idx(i, j): 1, b_idx(i, j+2): 1}
        elif r == 6: # B10 + B11
            return {b_idx(i+2, j): 1, b_idx(i+2, j+2): 1}

    # Strassen 2x2 coefficients for s_0..s_6
    # s0: (x00+x11)(y00+y11)
    # s1: (x10+x11)y00
    # s2: x00(y01-y11)
    # s3: x11(y10-y00)
    # s4: (x00+x01)y11
    # s5: (x10-x00)(y00+y01)
    # s6: (x01-x11)(y10+y11)
    
    # For each r, and each s (0..6), we construct the term.
    
    for r in range(7):
        # Define the linear forms for x and y for this r
        # We need to combine them for each s
        
        # Pre-calculate x and y dicts for this r
        x00 = get_x(r, 0, 0)
        x01 = get_x(r, 0, 1)
        x10 = get_x(r, 1, 0)
        x11 = get_x(r, 1, 1)
        
        y00 = get_y(r, 0, 0)
        y01 = get_y(r, 0, 1)
        y10 = get_y(r, 1, 0)
        y11 = get_y(r, 1, 1)
        
        # Define the 7 terms
        # Each term is a pair (A_coeffs, B_coeffs)
        # A_coeffs is a dict {a_idx: coeff}
        # B_coeffs is a dict {b_idx: coeff}
        
        terms = []
        
        # s0
        a_s0 = add_blocks(x00, x11)
        b_s0 = add_blocks(y00, y11)
        terms.append((a_s0, b_s0))
        
        # s1
        a_s1 = add_blocks(x10, x11)
        b_s1 = y00
        terms.append((a_s1, b_s1))
        
        # s2
        a_s2 = x00
        b_s2 = sub_blocks(y01, y11)
        terms.append((a_s2, b_s2))
        
        # s3
        a_s3 = x11
        b_s3 = sub_blocks(y10, y00)
        terms.append((a_s3, b_s3))
        
        # s4
        a_s4 = add_blocks(x00, x01)
        b_s4 = y11
        terms.append((a_s4, b_s4))
        
        # s5
        a_s5 = sub_blocks(x10, x00)
        b_s5 = add_blocks(y00, y01)
        terms.append((a_s5, b_s5))
        
        # s6
        a_s6 = sub_blocks(x01, x11)
        b_s6 = add_blocks(y10, y11)
        terms.append((a_s6, b_s6))
        
        for s in range(7):
            t = r * 7 + s
            a_coeffs, b_coeffs = terms[s]
            
            # Fill U
            for idx, coeff in a_coeffs.items():
                U[idx, t] = coeff
                
            # Fill V
            for idx, coeff in b_coeffs.items():
                V[idx, t] = coeff
                
            # Fill W
            # We need to determine how s contributes to C
            # M_r element contributions:
            # M00: s0 + s3 - s4 + s6
            # M01: s2 + s4
            # M10: s1 + s3
            # M11: s0 - s1 + s2 + s5
            
            c_m00 = 0
            c_m01 = 0
            c_m10 = 0
            c_m11 = 0
            
            if s == 0:
                c_m00 = 1
                c_m11 = 1
            elif s == 1:
                c_m10 = 1
                c_m11 = -1
            elif s == 2:
                c_m01 = 1
                c_m11 = 1
            elif s == 3:
                c_m00 = 1
                c_m10 = 1
            elif s == 4:
                c_m00 = -1
                c_m01 = 1
            elif s == 5:
                c_m11 = 1
            elif s == 6:
                c_m00 = 1
            
            # Now map M_r to C blocks
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # r=0 is M1
            # r=1 is M2
            # ...
            
            c_c00 = 0
            c_c01 = 0
            c_c10 = 0
            c_c11 = 0
            
            if r == 0: # M1
                c_c00 = 1
                c_c11 = 1
            elif r == 1: # M2
                c_c10 = 1
                c_c11 = -1
            elif r == 2: # M3
                c_c01 = 1
                c_c11 = 1
            elif r == 3: # M4
                c_c00 = 1
                c_c10 = 1
            elif r == 4: # M5
                c_c00 = -1
                c_c01 = 1
            elif r == 5: # M6
                c_c11 = 1
            elif r == 6: # M7
                c_c00 = 1
            
            # Combine
            # C element (i, j) gets contribution from M_r element (block_i, block_j)
            # C00 block is 2x2. C00[i,j] += c_c00 * M_r[bi, bj]
            # M_r[bi, bj] is determined by c_m...
            
            # C00 block indices: i in {0,1}, j in {0,1}
            # M_r[0,0] corresponds to c_m00
            # M_r[0,1] corresponds to c_m01
            # M_r[1,0] corresponds to c_m10
            # M_r[1,1] corresponds to c_m11
            
            # C00[0,0] += c_c00 * c_m00
            # C00[0,1] += c_c00 * c_m01
            # C00[1,0] += c_c00 * c_m10
            # C00[1,1] += c_c00 * c_m11
            
            # C01 block indices: i in {0,1}, j in {2,3}
            # C01[0,0] is C[0,2].
            
            # Let's iterate over the 4 elements of the 2x2 block
            for bi in range(2):
                for bj in range(2):
                    # Get coefficient for M_r[bi, bj]
                    if bi == 0 and bj == 0:
                        m_coeff = c_m00
                    elif bi == 0 and bj == 1:
                        m_coeff = c_m01
                    elif bi == 1 and bj == 0:
                        m_coeff = c_m10
                    else:
                        m_coeff = c_m11
                    
                    if m_coeff == 0:
                        continue
                        
                    # Which C block?
                    if r == 0:
                        # M1 contributes to C00 and C11
                        if c_c00 != 0:
                            c_block_i, c_block_j = 0, 0
                            c_total = c_c00 * m_coeff
                            c_idx_val = c_idx(c_block_i*2 + bi, c_block_j*2 + bj)
                            W[c_idx_val, t] += c_total
                        if c_c11 != 0:
                            c_block_i, c_block_j = 1, 1
                            c_total = c_c11 * m_coeff
                            c_idx_val = c_idx(c_block_i*2 + bi, c_block_j*2 + bj)
                            W[c_idx_val, t] += c_total
                    elif r == 1:
                        # M2 contributes to C10 and C11
                        if c_c10 != 0:
                            c_block_i, c_block_j = 1, 0
                            c_total = c_c10 * m_coeff
                            c_idx_val = c_idx(c_block_i*2 + bi, c_block_j*2 + bj)
                            W[c_idx_val, t] += c_total
                        if c_c11 != 0:
                            c_block_i, c_block_j = 1, 1
                            c_total = c_c11 * m_coeff
                            c_idx_val = c_idx(c_block_i*2 + bi, c_block_j*2 + bj)
                            W[c_idx_val, t] += c_total
                    elif r == 2:
                        # M3 contributes to C01 and C11
                        if c_c01 != 0:
                            c_block_i, c_block_j = 0, 1
                            c_total = c_c01 * m_coeff
                            c_idx_val = c_idx(c_block_i*2 + bi, c_block_j*2 + bj)
                            W[c_idx_val, t] += c_total
                        if c_c11 != 0:
                            c_block_i, c_block_j = 1, 1
                            c_total = c_c11 * m_coeff
                            c_idx_val = c_idx(c_block_i*2 + bi, c_block_j*2 + bj)
                            W[c_idx_val, t] += c_total
                    elif r == 3:
                        # M4 contributes to C00 and C10
                        if c_c00 != 0:
                            c_block_i, c_block_j = 0, 0
                            c_total = c_c00 * m_coeff
                            c_idx_val = c_idx(c_block_i*2 + bi, c_block_j*2 + bj)
                            W[c_idx_val, t] += c_total
                        if c_c10 != 0:
                            c_block_i, c_block_j = 1, 0
                            c_total = c_c10 * m_coeff
                            c_idx_val = c_idx(c_block_i*2 + bi, c_block_j*2 + bj)
                            W[c_idx_val, t] += c_total
                    elif r == 4:
                        # M5 contributes to C00 and C01
                        if c_c00 != 0:
                            c_block_i, c_block_j = 0, 0
                            c_total = c_c00 * m_coeff
                            c_idx_val = c_idx(c_block_i*2 + bi, c_block_j*2 + bj)
                            W[c_idx_val, t] += c_total
                        if c_c01 != 0:
                            c_block_i, c_block_j = 0, 1
                            c_total = c_c01 * m_coeff
                            c_idx_val = c_idx(c_block_i*2 + bi, c_block_j*2 + bj)
                            W[c_idx_val, t] += c_total
                    elif r == 5:
                        # M6 contributes to C11
                        if c_c11 != 0:
                            c_block_i, c_block_j = 1, 1
                            c_total = c_c11 * m_coeff
                            c_idx_val = c_idx(c_block_i*2 + bi, c_block_j*2 + bj)
                            W[c_idx_val, t] += c_total
                    elif r == 6:
                        # M7 contributes to C00
                        if c_c00 != 0:
                            c_block_i, c_block_j = 0, 0
                            c_total = c_c00 * m_coeff
                            c_idx_val = c_idx(c_block_i*2 + bi, c_block_j*2 + bj)
                            W[c_idx_val, t] += c_total

    return {'U': U, 'V': V, 'W': W}
