import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    # U shape: (n*m, rank) = (20, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (24, rank)
    
    # We will build the decomposition block by block.
    # Total rank is 108.
    # We use a list to store columns, then convert to matrix.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, block_type):
        # a_row_start, a_col_start: top-left of A block
        # b_row_start, b_col_start: top-left of B block
        # c_row_start, c_col_start: top-left of C block
        # block_type: 'strassen' or 'schoolbook_212'
        
        if block_type == 'strassen':
            # 2x2x2 block
            # A local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # B local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # C local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            
            # Global row indices for A
            a_rows = [
                a_row_start * m + a_col_start,     # 0,0
                a_row_start * m + a_col_start + 1, # 0,1
                (a_row_start + 1) * m + a_col_start,     # 1,0
                (a_row_start + 1) * m + a_col_start + 1  # 1,1
            ]
            # Global row indices for B
            b_rows = [
                b_row_start * p + b_col_start,     # 0,0
                b_row_start * p + b_col_start + 1, # 0,1
                (b_row_start + 1) * p + b_col_start,     # 1,0
                (b_row_start + 1) * p + b_col_start + 1  # 1,1
            ]
            # Global row indices for C (W)
            # W index is k*n + i. Here k is row of C, i is col of C?
            # Task says W[k*n+i, r]. C is p x n.
            # So C[row, col] corresponds to W[row*n + col].
            c_rows = [
                c_row_start * n + c_col_start,     # 0,0
                c_row_start * n + c_col_start + 1, # 0,1
                (c_row_start + 1) * n + c_col_start,     # 1,0
                (c_row_start + 1) * n + c_col_start + 1  # 1,1
            ]
            
            # Strassen terms
            # m1 = (a+d)(e+h)
            u1 = {a_rows[0]: 1, a_rows[3]: 1}
            v1 = {b_rows[0]: 1, b_rows[3]: 1}
            w1 = {c_rows[0]: 1, c_rows[3]: 1}
            
            # m2 = (c+d)e
            u2 = {a_rows[2]: 1, a_rows[3]: 1}
            v2 = {b_rows[0]: 1}
            w2 = {c_rows[2]: 1}
            
            # m3 = a(f-h)
            u3 = {a_rows[0]: 1}
            v3 = {b_rows[1]: 1, b_rows[3]: -1}
            w3 = {c_rows[1]: 1, c_rows[3]: 1}
            
            # m4 = d(g-e)
            u4 = {a_rows[3]: 1}
            v4 = {b_rows[2]: 1, b_rows[0]: -1}
            w4 = {c_rows[0]: 1, c_rows[2]: 1}
            
            # m5 = (a+b)h
            u5 = {a_rows[0]: 1, a_rows[1]: 1}
            v5 = {b_rows[3]: 1}
            w5 = {c_rows[0]: -1, c_rows[1]: 1}
            
            # m6 = (c-a)(e+f)
            u6 = {a_rows[0]: -1, a_rows[2]: 1}
            v6 = {b_rows[0]: 1, b_rows[1]: 1}
            w6 = {c_rows[3]: 1}
            
            # m7 = (b-d)(g+h)
            u7 = {a_rows[1]: 1, a_rows[3]: -1}
            v7 = {b_rows[2]: 1, b_rows[3]: 1}
            w7 = {c_rows[0]: 1, c_rows[3]: 1}
            
            terms = [
                (u1, v1, w1),
                (u2, v2, w2),
                (u3, v3, w3),
                (u4, v4, w4),
                (u5, v5, w5),
                (u6, v6, w6),
                (u7, v7, w7)
            ]
            
        elif block_type == 'schoolbook_212':
            # 2x1x2 block
            # A local: 0:(0,0), 1:(1,0)
            # B local: 0:(0,0), 1:(0,1)
            # C local: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            
            a_rows = [
                a_row_start * m + a_col_start,     # 0,0
                (a_row_start + 1) * m + a_col_start  # 1,0
            ]
            b_rows = [
                b_row_start * p + b_col_start,     # 0,0
                b_row_start * p + b_col_start + 1  # 0,1
            ]
            c_rows = [
                c_row_start * n + c_col_start,     # 0,0
                c_row_start * n + c_col_start + 1, # 0,1
                (c_row_start + 1) * n + c_col_start,     # 1,0
                (c_row_start + 1) * n + c_col_start + 1  # 1,1
            ]
            
            # ae
            u1 = {a_rows[0]: 1}
            v1 = {b_rows[0]: 1}
            w1 = {c_rows[0]: 1}
            
            # af
            u2 = {a_rows[0]: 1}
            v2 = {b_rows[1]: 1}
            w2 = {c_rows[1]: 1}
            
            # ce
            u3 = {a_rows[1]: 1}
            v3 = {b_rows[0]: 1}
            w3 = {c_rows[2]: 1}
            
            # cf
            u4 = {a_rows[1]: 1}
            v4 = {b_rows[1]: 1}
            w4 = {c_rows[3]: 1}
            
            terms = [
                (u1, v1, w1),
                (u2, v2, w2),
                (u3, v3, w3),
                (u4, v4, w4)
            ]
        else:
            return

        for u, v, w in terms:
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            
            for idx, val in u.items():
                U_col[idx] = val
            for idx, val in v.items():
                V_col[idx] = val
            for idx, val in w.items():
                W_col[idx] = val
                
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # Define blocks
    # A split: rows 0-1, 2-3. Cols 0-1, 2-3, 4.
    # B split: rows 0-1, 2-3, 4. Cols 0-1, 2-3, 4-5.
    
    # Strassen blocks (2x2x2)
    # A row pairs: (0,1), (2,3) -> start 0, 2
    # B col pairs: (0,1), (2,3), (4,5) -> start 0, 2, 4
    # B row pairs: (0,1), (2,3) -> start 0, 2
    # C row pairs: (0,1), (2,3) -> start 0, 2
    # C col pairs: (0,1), (2,3), (4,5) -> start 0, 2, 4
    
    a_row_starts = [0, 2]
    b_row_starts = [0, 2]
    a_col_starts_strassen = [0, 2] # for the first two middle dims
    b_col_starts = [0, 2, 4]
    c_row_starts = [0, 2]
    c_col_starts = [0, 2, 4]
    
    # 12 Strassen blocks
    for ar in a_row_starts:
        for br in b_row_starts:
            for ac in a_col_starts_strassen:
                for bc in b_col_starts:
                    cr = ar
                    cc = bc
                    add_block(ar, ac, br, bc, cr, cc, 'strassen')
                    
    # Schoolbook blocks (2x1x2)
    # A col start: 4
    # B row start: 4
    a_col_start_sb = 4
    b_row_start_sb = 4
    
    # 6 Schoolbook blocks
    for ar in a_row_starts:
        for br in [4]:
            for bc in b_col_starts:
                cr = ar
                cc = bc
                add_block(ar, a_col_start_sb, br, bc, cr, cc, 'schoolbook_212')

    rank = len(U_cols)
    
    # Convert to matrices
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

