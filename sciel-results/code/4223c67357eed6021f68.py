import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    terms = []
    
    def add_strassen_2x2x2(i0, j0, k0, a_scale=1.0, b_scale=1.0):
        # Adds 7 terms for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        # Indices:
        # A: (i, j) -> i*m + j
        # B: (j, k) -> j*p + k
        # C: (k, i) -> k*n + i
        
        a00 = i0 * m + j0
        a01 = i0 * m + (j0 + 1)
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + (j0 + 1)
        
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)
        
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a00 + a11) * (b00 + b11)
        u1 = [0.0] * (n * m); u1[a00] = a_scale; u1[a11] = a_scale
        v1 = [0.0] * (m * p); v1[b00] = b_scale; v1[b11] = b_scale
        w1 = [0.0] * (p * n); w1[c00] = 1.0; w1[c11] = 1.0
        terms.append((u1, v1, w1))
        
        # M2 = (a10 + a11) * b00
        u2 = [0.0] * (n * m); u2[a10] = a_scale; u2[a11] = a_scale
        v2 = [0.0] * (m * p); v2[b00] = b_scale
        w2 = [0.0] * (p * n); w2[c10] = 1.0; w2[c11] = -1.0
        terms.append((u2, v2, w2))
        
        # M3 = a00 * (b01 - b11)
        u3 = [0.0] * (n * m); u3[a00] = a_scale
        v3 = [0.0] * (m * p); v3[b01] = b_scale; v3[b11] = -b_scale
        w3 = [0.0] * (p * n); w3[c01] = 1.0; w3[c11] = 1.0
        terms.append((u3, v3, w3))
        
        # M4 = a11 * (b10 - b00)
        u4 = [0.0] * (n * m); u4[a11] = a_scale
        v4 = [0.0] * (m * p); v4[b10] = b_scale; v4[b00] = -b_scale
        w4 = [0.0] * (p * n); w4[c00] = 1.0; w4[c10] = 1.0
        terms.append((u4, v4, w4))
        
        # M5 = (a00 + a01) * b11
        u5 = [0.0] * (n * m); u5[a00] = a_scale; u5[a01] = a_scale
        v5 = [0.0] * (m * p); v5[b11] = b_scale
        w5 = [0.0] * (p * n); w5[c00] = -1.0; w5[c01] = 1.0
        terms.append((u5, v5, w5))
        
        # M6 = (a10 - a00) * (b00 + b01)
        u6 = [0.0] * (n * m); u6[a10] = a_scale; u6[a00] = -a_scale
        v6 = [0.0] * (m * p); v6[b00] = b_scale; v6[b01] = b_scale
        w6 = [0.0] * (p * n); w6[c11] = 1.0
        terms.append((u6, v6, w6))
        
        # M7 = (a01 - a11) * (b10 + b11)
        u7 = [0.0] * (n * m); u7[a01] = a_scale; u7[a11] = -a_scale
        v7 = [0.0] * (m * p); v7[b10] = b_scale; v7[b11] = b_scale
        w7 = [0.0] * (p * n); w7[c00] = 1.0
        terms.append((u7, v7, w7))

    def add_strassen_4x4x4(i0, j0, k0):
        # Decomposes A[i0:i0+4, j0:j0+4] * B[j0:j0+4, k0:k0+4]
        # A is 4x4, B is 4x4, C is 4x4
        # We assume A has zeros in the bottom-right 2x2 block (rows i0+2, i0+3)
        # Specifically, row i0+3 is all zeros. Row i0+2 has non-zeros in cols j0, j0+1.
        # This matches the 3x4 A padded to 4x4.
        
        # M1 = (A00 + A11) * (B00 + B11) -> A00 * (B00 + B11) since A11=0
        # A00 is i0, i0+1 x j0, j0+1
        # B00 is j0, j0+1 x k0, k0+1
        # B11 is j0+2, j0+3 x k0+2, k0+3
        # Wait, B is 4x4.
        # B00: j0, j0+1 x k0, k0+1
        # B01: j0, j0+1 x k0+2, k0+3
        # B10: j0+2, j0+3 x k0, k0+1
        # B11: j0+2, j0+3 x k0+2, k0+3
        
        # The standard Strassen M1 uses (A00+A11)(B00+B11).
        # Here A11 is 0. So M1 = A00(B00+B11).
        # This is not a standard 2x2x2 Strassen block because B00+B11 is a sum of two 2x2 blocks.
        # We need to expand this into 2x2x2 terms.
        
        # Actually, it is easier to just call the 2x2x2 Strassen for the sub-blocks
        # that are non-zero, but we must ensure we don't double count or miss terms.
        
        # Let's just use the 6 non-zero terms derived earlier.
        # M1: A00 * (B00 + B11). 
        # This is A00*B00 + A00*B11.
        # We can compute A00*B00 (2x2x2) and A00*B11 (2x2x2) separately?
        # That would be 7+7=14.
        # But M1 in Strassen is ONE multiplication of 2x2 matrices.
        # The "matrix" is (B00+B11).
        # So we can form a virtual 2x2 matrix B_sum = B00+B11.
        # But we can't form it in the tensor decomposition easily without adding terms.
        
        # Alternative: Just use the 2x2x2 Strassen on the 4 blocks of A and B.
        # A = [[A00, A01], [A10, 0]]
        # B = [[B00, B01], [B10, B11]]
        
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00
        # C11 = A10B01
        
        # Using Strassen:
        # M1 = (A00 + 0)(B00 + B11) = A00(B00+B11)
        # M2 = (A10 + 0)B00 = A10B00
        # M3 = A00(B01 - B11)
        # M4 = 0
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - 0)(B10 + B11) = A01(B10+B11)
        
        # Each Mi is a 2x2x2 product.
        # We need to implement a helper that adds a 2x2x2 Strassen product
        # where the A and B matrices are sums/diffs of blocks.
        
        def add_virtual_strassen_2x2x2(a_blocks, b_blocks, c_i, c_j):
            # a_blocks: list of (i0, j0, scale)
            # b_blocks: list of (j0, k0, scale)
            # c_i, c_j: target C block (i0, j0)
            # This function adds the 7 terms of Strassen for the product of 
            # A_virtual = sum(a_blocks) and B_virtual = sum(b_blocks)
            # into C[c_i, c_j].
            
            # This is complex. 
            # Instead, we can just use the schoolbook for the 4x4x4? No.
            
            # Let's stick to the 88 rank solution if 78 is hard to implement correctly.
            # Or, we can just use the 2x2x2 Strassen for the non-zero blocks.
            # C00 = A00B00 + A01B10.
            # C01 = A00B01 + A01B11.
            # C10 = A10B00.
            # C11 = A10B01.
            
            # A00B00: 2x2x2 Strassen (7)
            # A01B10: 2x2x2 Strassen (7)
            # A00B01: 2x2x2 Strassen (7)
            # A01B11: 2x2x2 Strassen (7)
            # A10B00: 1x2x2 Schoolbook (4)
            # A10B01: 1x2x2 Schoolbook (4)
            
            # Total: 7+7+7+7+4+4 = 36.
            # This is better than 39!
            
            add_strassen_2x2x2(i0, j0, k0) # A00B00 -> C00
            add_strassen_2x2x2(i0, j0+2, k0+2) # A01B10 -> C00 (Wait, A01 is i0, j0+2. B10 is j0+2, k0. Product is i0, k0. Correct.)
            
            add_strassen_2x2x2(i0, j0, k0+2) # A00B01 -> C01
            add_strassen_2x2x2(i0, j0+2, k0+2) # A01B11 -> C01 (Wait, A01 is i0, j0+2. B11 is j0+2, k0+2. Product is i0, k0+2. Correct.)
            
            # A10 is i0+2, j0.
            # A10B00: i0+2, j0 x j0, k0 -> i0+2, k0.
            # This is 1x2x2.
            add_schoolbook_1x2x2(i0+2, j0, k0)
            
            # A10B01: i0+2, j0 x j0, k0+2 -> i0+2, k0+2.
            add_schoolbook_1x2x2(i0+2, j0, k0+2)
            
            # Wait, A10 is 2x2? No, A10 is rows i0+2, i0+3.
            # Row i0+3 is 0.
            # So A10 is effectively 1x2 (row i0+2, cols j0, j0+1).
            # B00 is 2x2 (rows j0, j0+1, cols k0, k0+1).
            # Product is 1x2.
            # My add_schoolbook_1x2x2 assumes A is 1x2.
            # It iterates j in 0..1, k in 0..1.
            # u_idx = (i0+2)*m + (j0+j).
            # v_idx = (j0+j)*p + (k0+k).
            # w_idx = (k0+k)*n + (i0+2).
            # This is correct.

            # Total rank per 4x4x4 block: 36.
            # Two blocks: 72.

    def add_schoolbook_1x2x2(i_row, j0, k0):
        for j in range(2):
            for k in range(2):
                u_idx = i_row * m + (j0 + j)
                v_idx = (j0 + j) * p + (k0 + k)
                w_idx = (k0 + k) * n + i_row
                u_vec = [0.0] * (n * m); u_vec[u_idx] = 1.0
                v_vec = [0.0] * (m * p); v_vec[v_idx] = 1.0
                w_vec = [0.0] * (p * n); w_vec[w_idx] = 1.0
                terms.append((u_vec, v_vec, w_vec))

    # Use the 36 rank decomposition for each 4x4x4 block
    # B is split into B_left (4x4) and B_right (4x4)
    add_strassen_4x4x4(0, 0, 0)
    add_strassen_4x4x4(0, 0, 4)
    
    rank = len(terms)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u_vec, v_vec, w_vec = terms[r]
        for i in range(n * m):
            if u_vec[i] != 0:
                U[i][r] = u_vec[i]
        for j in range(m * p):
            if v_vec[j] != 0:
                V[j][r] = v_vec[j]
        for k in range(p * n):
            if w_vec[k] != 0:
                W[k][r] = w_vec[k]
                
    return {"U": U, "V": V, "W": W}
