import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    terms = []  # list of (u, v, w)
    
    def add_term(u_dict, v_dict, w_dict):
        """u_dict, v_dict, w_dict are dicts mapping index -> value."""
        u = [0]*9
        v = [0]*12
        w = [0]*12
        for idx, val in u_dict.items():
            u[idx] += val
        for idx, val in v_dict.items():
            v[idx] += val
        for idx, val in w_dict.items():
            w[idx] += val
        terms.append((u, v, w))
    
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """Add 7 Strassen terms for a 2x2x2 block.
        A block at rows a_row_offset:a_row_offset+2, cols a_col_offset:a_col_offset+2
        B block at rows b_row_offset:b_row_offset+2, cols b_col_offset:b_col_offset+2
        C block at rows c_row_offset:c_row_offset+2, cols c_col_offset:c_col_offset+2
        """
        # M1 = (A00+A11)(B00+B11)
        add_term(
            {(a_row_offset)*3 + a_col_offset: 1, (a_row_offset+1)*3 + (a_col_offset+1): 1},
            {(b_row_offset)*4 + b_col_offset: 1, (b_row_offset+1)*4 + (b_col_offset+1): 1},
            {(b_col_offset)*3 + c_row_offset: 1, (b_col_offset+1)*3 + (c_row_offset+1): 1}
        )
        # M2 = (A10+A11)B00
        add_term(
            {(a_row_offset+1)*3 + a_col_offset: 1, (a_row_offset+1)*3 + (a_col_offset+1): 1},
            {(b_row_offset)*4 + b_col_offset: 1},
            {(b_col_offset)*3 + (c_row_offset+1): 1, (b_col_offset+1)*3 + (c_row_offset+1): -1}
        )
        # M3 = A00(B10-B11)
        add_term(
            {(a_row_offset)*3 + a_col_offset: 1},
            {(b_row_offset+1)*4 + b_col_offset: 1, (b_row_offset+1)*4 + (b_col_offset+1): -1},
            {(b_col_offset)*3 + (c_row_offset+1): 1, (b_col_offset+1)*3 + (c_row_offset+1): 1}
        )
        # M4 = A11(B01-B10)
        add_term(
            {(a_row_offset+1)*3 + (a_col_offset+1): 1},
            {(b_row_offset)*4 + (b_col_offset+1): 1, (b_row_offset+1)*4 + b_col_offset: -1},
            {(b_col_offset)*3 + c_row_offset: 1, (b_col_offset+1)*3 + c_row_offset: 1}
        )
        # M5 = (A00+A10)B11
        add_term(
            {(a_row_offset)*3 + a_col_offset: 1, (a_row_offset+1)*3 + a_col_offset: 1},
            {(b_row_offset+1)*4 + (b_col_offset+1): 1},
            {(b_col_offset)*3 + c_row_offset: -1, (b_col_offset+1)*3 + (c_row_offset+1): 1}
        )
        # M6 = (A01-A10)(B00+B10)
        add_term(
            {(a_row_offset)*3 + (a_col_offset+1): 1, (a_row_offset+1)*3 + a_col_offset: -1},
            {(b_row_offset)*4 + b_col_offset: 1, (b_row_offset+1)*4 + b_col_offset: 1},
            {(b_col_offset+1)*3 + (c_row_offset+1): 1}
        )
        # M7 = (A01-A11)(B01+B11)
        add_term(
            {(a_row_offset)*3 + (a_col_offset+1): 1, (a_row_offset+1)*3 + (a_col_offset+1): -1},
            {(b_row_offset)*4 + (b_col_offset+1): 1, (b_row_offset+1)*4 + (b_col_offset+1): 1},
            {(b_col_offset)*3 + c_row_offset: 1}
        )
    
    def add_schoolbook(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add schoolbook terms for a sub-block."""
        for i in a_rows:
            for j in a_cols:
                for k in b_cols:
                    # A[i,j] * B[j,k] → C[i, k-b_cols[0]+c_cols[0]]... 
                    # Actually, B row index is j, col index is k
                    # C row is i, col is k
                    add_term(
                        {i*3+j: 1},
                        {j*4+k: 1},
                        {k*3+i: 1}
                    )
    
    # A11*B11: Strassen
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # A12*B21: (2×1)·(1×2) → C11
    # A rows 0,1 col 2; B row 2 cols 0,1; C rows 0,1 cols 0,1
    for i in range(2):
        for k in range(2):
            add_term({i*3+2: 1}, {2*4+k: 1}, {k*3+i: 1})
    
    # A11*B12: Strassen
    add_strassen(0, 0, 0, 2, 0, 2)
    
    # A12*B22: (2×1)·(1×2) → C12
    # A rows 0,1 col 2; B row 2 cols 2,3; C rows 0,1 cols 2,3
    for i in range(2):
        for k in range(2,4):
            add_term({i*3+2: 1}, {2*4+k: 1}, {k*3+i: 1})
    
    # A21*B11: (1×2)·(2×2) → C21
    # A row 2 cols 0,1; B rows 0,1 cols 0,1; C row 2 cols 0,1
    for j in range(2):
        for k in range(2):
            add_term({2*3+j: 1}, {j*4+k: 1}, {k*3+2: 1})
    
    # A22*B21: (1×1)·(1×2) → C21
    # A row 2 col 2; B row 2 cols 0,1; C row 2 cols 0,1
    for k in range(2):
        add_term({2*3+2: 1}, {2*4+k: 1}, {k*3+2: 1})
    
    # A21*B12: (1×2)·(2×2) → C22
    # A row 2 cols 0,1; B rows 0,1 cols 2,3; C row 2 cols 2,3
    for j in range(2):
        for k in range(2,4):
            add_term({2*3+j: 1}, {j*4+k: 1}, {k*3+2: 1})
    
    # A22*B22: (1×1)·(1×2) → C22
    # A row 2 col 2; B row 2 cols 2,3; C row 2 cols 2,3
    for k in range(2,4):
        add_term({2*3+2: 1}, {2*4+k: 1}, {k*3+2: 1})
    
    rank = len(terms)
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(12)]
    W = [[0]*rank for _ in range(12)]
    for r, (u, v, w) in enumerate(terms):
        for idx in range(9):
            U[idx][r] = u[idx]
        for idx in range(12):
            V[idx][r] = v[idx]
        for idx in range(12):
            W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
