DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Strassen (2,2,2) decomposition
    # U: for A entries a=A[i1,j1], b=A[i1,j2], c=A[i2,j1], d=A[i2,j2]
    strassen_U = [
        [1, 0, 1, 0, 1, 1, 0],   # a
        [0, 0, 0, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, -1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]   # d
    ]
    strassen_V = [
        [1, 1, 0, 0, 0, 1, 0],   # w
        [0, 0, 1, 0, 0, 1, 0],   # x
        [0, 0, 0, 1, 0, 0, 1],   # y
        [1, 0, -1, -1, 1, 0, 1]  # z
    ]
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],  # C[i1,k1]
        [0, 1, 0, 1, 0, 0, 0],    # C[i2,k1]
        [0, 0, 1, 0, 1, 0, -1],   # C[i1,k2]
        [1, -1, 1, 0, 0, 1, 0]    # C[i2,k2]
    ]
    
    # (1,2,2) decomposition: A has 1 row, B has 2 rows 2 cols
    # C[0,k] = A[0]B[0,k] + A[1]B[1,k]
    # U: for A entries (1 row, 2 cols): positions j1, j2
    l22_U = [
        [1, 0],  # A[0] (j1)
        [0, 1]   # A[1] (j2)
    ]
    l22_V = [
        [1, 0],  # B[0,j1]
        [1, 0],  # B[0,j2]
        [0, 1],  # B[1,j1]
        [0, 1]   # B[1,j2]
    ]
    l22_W = [
        [1, 1],  # C[0,j1]
        [1, 1]   # C[0,j2]
    ]
    
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    def add_terms(u_entries, v_entries, w_entries):
        """u_entries: list of (u_row, value) for non-zero U entries in this term
           v_entries: list of (v_row, value) for non-zero V entries
           w_entries: list of (w_row, value) for non-zero W entries"""
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        for (row, val) in u_entries:
            U_row[row] = val
        for (row, val) in v_entries:
            V_row[row] = val
        for (row, val) in w_entries:
            W_row[row] = val
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
    
    # Partition definitions
    # A rows: groups [(0,1)] size 2, [2] size 1
    # A cols: groups [(0,1)] size 2, [(2,3)] size 2
    # B rows: groups [(0,1)] size 2, [(2,3)] size 2
    # B cols: groups [(0,1)], [(2,3)], [(4,5)], [(6,7)] size 2 each
    
    a_row_groups = [((0,1), 'strassen'), (2, 'single')]  # (indices, type)
    a_col_groups = [(0,1), (2,3)]
    b_row_groups = [(0,1), (2,3)]
    b_col_groups = [(0,1), (2,3), (4,5), (6,7)]
    
    for (a_rows, a_type) in a_row_groups:
        if a_type == 'strassen':
            i1, i2 = a_rows
        else:
            i1 = a_rows  # single row
            i2 = None
        
        for (j1, j2) in a_col_groups:
            for (bj1, bj2) in b_row_groups:
                for (k1, k2) in b_col_groups:
                    if a_type == 'strassen':
                        # (2,2,2) Strassen block
                        for t in range(7):
                            u_e = []
                            v_e = []
                            w_e = []
                            # U entries for a,b,c,d
                            u_e.append((i1*m + j1, strassen_U[0][t]))
                            u_e.append((i1*m + j2, strassen_U[1][t]))
                            u_e.append((i2*m + j1, strassen_U[2][t]))
                            u_e.append((i2*m + j2, strassen_U[3][t]))
                            # V entries for w,x,y,z
                            v_e.append((bj1*p + k1, strassen_V[0][t]))
                            v_e.append((bj1*p + k2, strassen_V[1][t]))
                            v_e.append((bj2*p + k1, strassen_V[2][t]))
                            v_e.append((bj2*p + k2, strassen_V[3][t]))
                            # W entries for C[i1,k1], C[i2,k1], C[i1,k2], C[i2,k2]
                            w_e.append((k1*n + i1, strassen_W[0][t]))
                            w_e.append((k1*n + i2, strassen_W[1][t]))
                            w_e.append((k2*n + i1, strassen_W[2][t]))
                            w_e.append((k2*n + i2, strassen_W[3][t]))
                            add_terms(u_e, v_e, w_e)
                    else:
                        # (1,2,2) block: single row i1
                        for t in range(2):
                            u_e = []
                            v_e = []
                            w_e = []
                            u_e.append((i1*m + j1, l22_U[0][t]))
                            u_e.append((i1*m + j2, l22_U[1][t]))
                            v_e.append((bj1*p + k1, l22_V[0][t]))
                            v_e.append((bj1*p + k2, l22_V[1][t]))
                            v_e.append((bj2*p + k1, l22_V[2][t]))
                            v_e.append((bj2*p + k2, l22_V[3][t]))
                            w_e.append((k1*n + i1, l22_W[0][t]))
                            w_e.append((k2*n + i1, l22_W[1][t]))
                            add_terms(u_e, v_e, w_e)
    
    U = U_list
    V = V_list
    W = W_list
    
    return {"U": U, "V": V, "W": W}
