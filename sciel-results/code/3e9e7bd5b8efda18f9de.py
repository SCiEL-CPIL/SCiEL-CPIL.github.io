import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication.
    Constructed directly via tensor identity:
    For each (i, k, j), sum_r U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r] = 1
    (using trace convention W[k*n+i, r] => here W is indexed by (j, i) as j*4+i for the C[j, i] slot? 
     Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]".
     Let's re-read the current code's W indexing.
     Current code: w_idx(gr, gc) = gr * 4 + gc?
     No, current code says: `def w_idx(i, j): return j * 4 + i # Trace convention: W[k*n+i, r] -> W[col*n + row]`
     So if W corresponds to C[row, col], then W_index = col * n + row.
     In the code: gr is the row of C (0..3), gc is the col of C (0..3).
     w_i = w_idx(gr, gc) => gc * 4 + gr.
     So W[col*4+row, r].
     
     My implementation must match this.
     """
    
    n = 4
    rank = 49
    
    # Initialize factors
    U = np.zeros((n*n, rank), dtype=np.int64)
    V = np.zeros((n*n, rank), dtype=np.int64)
    W = np.zeros((n*n, rank), dtype=np.int64)
    
    # Helpers for combo manipulation
    def add_c(c1, c2):
        d = {}
        for i, v in c1: d[i] = d.get(i, 0) + v
        for i, v in c2: d[i] = d.get(i, 0) + v
        return [(k, v) for k, v in d.items() if v != 0]
        
    def sub_c(c1, c2):
        d = {}
        for i, v in c1: d[i] = d.get(i, 0) + v
        for i, v in c2: d[i] = d.get(i, 0) - v
        return [(k, v) for k, v in d.items() if v != 0]
        
    # Block element definitions (global indices for A and B)
    # A00: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # A01: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    # A10: (2,0), (2,1), (3,0), (3,1) -> 8, 9, 12, 13
    # A11: (2,2), (2,3), (3,2), (3,3) -> 10, 11, 14, 15
    
    A00 = [[(0,1)], [(1,1)], [(4,1)], [(5,1)]]
    A01 = [[(2,1)], [(3,1)], [(6,1)], [(7,1)]]
    A10 = [[(8,1)], [(9,1)], [(12,1)], [(13,1)]]
    A11 = [[(10,1)], [(11,1)], [(14,1)], [(15,1)]]
    
    B00 = [[(0,1)], [(1,1)], [(4,1)], [(5,1)]]
    B01 = [[(2,1)], [(3,1)], [(6,1)], [(7,1)]]
    B10 = [[(8,1)], [(9,1)], [(12,1)], [(13,1)]]
    B11 = [[(10,1)], [(11,1)], [(14,1)], [(15,1)]]
    
    # Block operations
    def block_add(M1, M2):
        return [add_c(M1[i], M2[i]) for i in range(4)]
        
    def block_sub(M1, M2):
        return [sub_c(M1[i], M2[i]) for i in range(4)]
        
    # Level 1 Strassen definitions (7 block products)
    # X, Y are 2x2 matrices of combos
    
    # T0: (A00+A11)(B00+B11)
    X0 = block_add(A00, A11)
    Y0 = block_add(B00, B11)
    
    # T1: (A10+A11)B00
    X1 = block_add(A10, A11)
    Y1 = B00
    
    # T2: A00(B01-B11)
    X2 = A00
    Y2 = block_sub(B01, B11)
    
    # T3: A11(B10-B00)
    X3 = A11
    Y3 = block_sub(B10, B00)
    
    # T4: (A00+A01)B11
    X4 = block_add(A00, A01)
    Y4 = B11
    
    # T5: (A10-A00)(B00+B01)
    X5 = block_sub(A10, A00)
    Y5 = block_add(B00, B01)
    
    # T6: (A01-A11)(B10+B11)
    X6 = block_sub(A01, A11)
    Y6 = block_add(B10, B11)
    
    X_list = [X0, X1, X2, X3, X4, X5, X6]
    Y_list = [Y0, Y1, Y2, Y3, Y4, Y5, Y6]
    
    # Block maps for Level 1 products to C blocks
    # block_maps[t] = [(c_row_block, c_col_block, coeff), ...]
    block_maps = [
        [(0, 0, 1), (1, 1, 1)],   # P0
        [(1, 0, 1), (1, 1, -1)],  # P1
        [(0, 1, 1), (1, 1, 1)],   # P2
        [(0, 0, 1), (1, 0, 1)],   # P3
        [(0, 0, -1), (0, 1, 1)],  # P4
        [(1, 1, 1)],              # P5
        [(0, 0, 1)]               # P6
    ]
    
    # Level 2 Strassen definitions for 2x2 matrices X, Y
    # Sub-contributions of M0..M6 to local Z (2x2 result)
    # Z00 = M0 + M3 - M4 + M6
    # Z01 = M2 + M4
    # Z10 = M1 + M3
    # Z11 = M0 - M1 + M2 + M5
    # Indices in Z: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    sub_contribs = [
        [(0, 1), (3, 1)],  # M0
        [(2, 1), (3, -1)], # M1
        [(1, 1), (3, 1)],  # M2
        [(0, 1), (2, 1)],  # M3
        [(0, -1), (1, 1)], # M4
        [(3, 1)],          # M5
        [(0, 1)]           # M6
    ]
    
    # Populate U, V, W
    for t in range(7):
        X = X_list[t]
        Y = Y_list[t]
        
        x00, x01, x10, x11 = X[0], X[1], X[2], X[3]
        y00, y01, y10, y11 = Y[0], Y[1], Y[2], Y[3]
        
        # Compute 7 Strassen products M0..M6 for the 2x2 level
        # M0: (x00+x11)(y00+y11)
        m0_l = add_c(x00, x11)
        m0_r = add_c(y00, y11)
        
        # M1: (x10+x11)y00
        m1_l = add_c(x10, x11)
        m1_r = y00
        
        # M2: x00(y01-y11)
        m2_l = x00
        m2_r = sub_c(y01, y11)
        
        # M3: x11(y10-y00)
        m3_l = x11
        m3_r = sub_c(y10, y00)
        
        # M4: (x00+x01)y11
        m4_l = add_c(x00, x01)
        m4_r = y11
        
        # M5: (x10-x00)(y00+y01)
        m5_l = sub_c(x10, x00)
        m5_r = add_c(y00, y01)
        
        # M6: (x01-x11)(y10+y11)
        m6_l = sub_c(x01, x11)
        m6_r = add_c(y10, y11)
        
        ms = [
            (m0_l, m0_r),
            (m1_l, m1_r),
            (m2_l, m2_r),
            (m3_l, m3_r),
            (m4_l, m4_r),
            (m5_l, m5_r),
            (m6_l, m6_r)
        ]
        
        for k in range(7):
            rank_idx = t * 7 + k
            l_combo, r_combo = ms[k]
            
            # Update U (A factors)
            for idx, coeff in l_combo:
                U[idx, rank_idx] += coeff
            
            # Update V (B factors)
            for idx, coeff in r_combo:
                V[idx, rank_idx] += coeff
            
            # Update W (C factors)
            # For each contribution of Mk to local Z elements
            for z_idx, coeff in sub_contribs[k]:
                # z_idx 0->Z00, 1->Z01, 2->Z10, 3->Z11
                lr = z_idx // 2
                lc = z_idx % 2
                
                # Map to global C blocks using block_maps[t]
                for c_br, c_bc, c_coeff in block_maps[t]:
                    # Global row index in C
                    gr = 2 * c_br + lr
                    # Global col index in C
                    gc = 2 * c_bc + lc
                    
                    # Trace convention for W: W[col*n + row]
                    w_i = gc * n + gr
                    W[w_i, rank_idx] += coeff * c_coeff
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
