import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen decomposition for 2x2x2
    # Local indices i,j,k in {0,1}
    # U_local[i,j], V_local[j,k], W_local[k,i]
    # We define the 7 terms as lists of (i, j, val) for U, etc.
    
    def add_strassen_block(offset, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # a_row_offset, a_col_offset: offset in A (2x4)
        # b_row_offset, b_col_offset: offset in B (4x4)
        # c_row_offset, c_col_offset: offset in C (2x4)
        
        # The 7 Strassen terms
        # 1. (A00+A11)(B00+B11) -> C00, C11
        # 2. (A10+A11)B00 -> C11
        # 3. A00(B10-B11) -> C01
        # 4. A11(B01-B00) -> C00, C01
        # 5. (A00+A10)B11 -> C01
        # 6. (A10-A00)(B00+B01) -> C10
        # 7. (A01-A11)(B10+B11) -> C00, C10
        
        # We need to map local (i,j) to global A index.
        # A is 2x4. Local A is 2x2.
        # Global A index: row = a_row_offset + i, col = a_col_offset + j.
        # Global A flat index: row * m + col.
        
        # Global B index: row = b_row_offset + j, col = b_col_offset + k.
        # Global B flat index: row * p + col.
        
        # Global C index: row = c_row_offset + i, col = c_col_offset + k.
        # Global C flat index: row * n + col? No, W is p*n.
        # W[k*n+i, r].
        # Global W index: k * n + i.
        # k is col of C, i is row of C.
        
        terms = []
        
        # Term 1
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11
        u = [(0, 0, 1), (1, 1, 1)]
        v = [(0, 0, 1), (1, 1, 1)]
        w = [(0, 0, 1), (1, 1, 1)]
        terms.append((u, v, w))
        
        # Term 2
        # U: A10, A11
        # V: B00
        # W: C11
        u = [(1, 0, 1), (1, 1, 1)]
        v = [(0, 0, 1)]
        w = [(1, 1, 1)]
        terms.append((u, v, w))
        
        # Term 3
        # U: A00
        # V: B10, B11
        # W: C01
        u = [(0, 0, 1)]
        v = [(1, 0, 1), (1, 1, -1)]
        w = [(0, 1, 1)]
        terms.append((u, v, w))
        
        # Term 4
        # U: A11
        # V: B01, B00
        # W: C00, C01
        u = [(1, 1, 1)]
        v = [(0, 1, 1), (0, 0, -1)]
        w = [(0, 0, 1), (0, 1, 1)]
        terms.append((u, v, w))
        
        # Term 5
        # U: A00, A10
        # V: B11
        # W: C01
        u = [(0, 0, 1), (1, 0, 1)]
        v = [(1, 1, 1)]
        w = [(0, 1, 1)]
        terms.append((u, v, w))
        
        # Term 6
        # U: A10, A00
        # V: B00, B01
        # W: C10
        u = [(1, 0, 1), (0, 0, -1)]
        v = [(0, 0, 1), (0, 1, 1)]
        w = [(1, 0, 1)]
        terms.append((u, v, w))
        
        # Term 7
        # U: A01, A11
        # V: B10, B11
        # W: C00, C10
        u = [(0, 1, 1), (1, 1, -1)]
        v = [(1, 0, 1), (1, 1, 1)]
        w = [(0, 0, 1), (1, 0, 1)]
        terms.append((u, v, w))
        
        for u_terms, v_terms, w_terms in terms:
            for (li, lj, val) in u_terms:
                global_row = a_row_offset + li
                global_col = a_col_offset + lj
                U[global_row * m + global_col, offset] += val
            for (lj, lk, val) in v_terms:
                global_row = b_row_offset + lj
                global_col = b_col_offset + lk
                V[global_row * p + global_col, offset] += val
            for (lk, li, val) in w_terms:
                global_row = c_row_offset + li
                global_col = c_col_offset + lk
                W[global_col * n + global_row, offset] += val
            offset += 1
        
        return offset

    offset = 0
    # P1: A00 B00 -> C00
    # A00 is A[0:2, 0:2]
    # B00 is B[0:2, 0:2]
    # C00 is C[0:2, 0:2]
    offset = add_strassen_block(offset, 0, 0, 0, 0, 0, 0)
    
    # P2: A01 B10 -> C00
    # A01 is A[0:2, 2:4]
    # B10 is B[2:4, 0:2]
    # C00 is C[0:2, 0:2]
    offset = add_strassen_block(offset, 0, 2, 2, 0, 0, 0)
    
    # P3: A00 B01 -> C01
    # A00 is A[0:2, 0:2]
    # B01 is B[0:2, 2:4]
    # C01 is C[0:2, 2:4]
    offset = add_strassen_block(offset, 0, 0, 0, 2, 0, 2)
    
    # P4: A01 B11 -> C01
    # A01 is A[0:2, 2:4]
    # B11 is B[2:4, 2:4]
    # C01 is C[0:2, 2:4]
    offset = add_strassen_block(offset, 0, 2, 2, 2, 0, 2)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
