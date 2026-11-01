"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We will build lists of columns for U, V, W
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: dict {idx: val} for U
        v_vec: dict {idx: val} for V
        w_vec: dict {idx: val} for W
        """
        u_col = [0] * (n * m)
        for idx, val in u_vec.items():
            u_col[idx] = val
        v_col = [0] * (m * p)
        for idx, val in v_vec.items():
            v_col[idx] = val
        w_col = [0] * (p * n)
        for idx, val in w_vec.items():
            w_col[idx] = val
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds Strassen decomposition for 2x2x2 block.
        A block starts at (a_row_offset, a_col_offset)
        B block starts at (b_row_offset, b_col_offset)
        C block starts at (c_row_offset, c_col_offset)
        """
        # A indices in global U: (r + a_row_offset) * m + (c + a_col_offset)
        # B indices in global V: (r + b_row_offset) * p + (c + b_col_offset)
        # C indices in global W: (c + c_col_offset) * n + (r + c_row_offset)
        
        # P1 = (A00 + A11)(B00 + B11)
        u1 = {
            (0 + a_row_offset) * m + (0 + a_col_offset): 1,
            (1 + a_row_offset) * m + (1 + a_col_offset): 1
        }
        v1 = {
            (0 + b_row_offset) * p + (0 + b_col_offset): 1,
            (1 + b_row_offset) * p + (1 + b_col_offset): 1
        }
        w1 = {
            (0 + c_col_offset) * n + (0 + c_row_offset): 1, # C00
            (1 + c_col_offset) * n + (1 + c_row_offset): 1  # C11
        }
        add_term(u1, v1, w1)
        
        # P2 = (A10 + A11) B00
        u2 = {
            (1 + a_row_offset) * m + (0 + a_col_offset): 1,
            (1 + a_row_offset) * m + (1 + a_col_offset): 1
        }
        v2 = {
            (0 + b_row_offset) * p + (0 + b_col_offset): 1
        }
        w2 = {
            (0 + c_col_offset) * n + (1 + c_row_offset): 1, # C10
            (1 + c_col_offset) * n + (1 + c_row_offset): -1  # C11
        }
        add_term(u2, v2, w2)
        
        # P3 = A00 (B01 - B11)
        u3 = {
            (0 + a_row_offset) * m + (0 + a_col_offset): 1
        }
        v3 = {
            (0 + b_row_offset) * p + (1 + b_col_offset): 1,
            (1 + b_row_offset) * p + (1 + b_col_offset): -1
        }
        w3 = {
            (1 + c_col_offset) * n + (0 + c_row_offset): 1, # C01
            (1 + c_col_offset) * n + (1 + c_row_offset): 1  # C11
        }
        add_term(u3, v3, w3)
        
        # P4 = A11 (B10 - B00)
        u4 = {
            (1 + a_row_offset) * m + (1 + a_col_offset): 1
        }
        v4 = {
            (1 + b_row_offset) * p + (0 + b_col_offset): 1,
            (0 + b_row_offset) * p + (0 + b_col_offset): -1
        }
        w4 = {
            (0 + c_col_offset) * n + (0 + c_row_offset): 1, # C00
            (0 + c_col_offset) * n + (1 + c_row_offset): 1  # C10
        }
        add_term(u4, v4, w4)
        
        # P5 = (A00 + A01) B11
        u5 = {
            (0 + a_row_offset) * m + (0 + a_col_offset): 1,
            (0 + a_row_offset) * m + (1 + a_col_offset): 1
        }
        v5 = {
            (1 + b_row_offset) * p + (1 + b_col_offset): 1
        }
        w5 = {
            (0 + c_col_offset) * n + (0 + c_row_offset): -1, # C00
            (1 + c_col_offset) * n + (0 + c_row_offset): 1   # C01
        }
        add_term(u5, v5, w5)
        
        # P6 = (A10 - A00)(B00 + B01)
        u6 = {
            (1 + a_row_offset) * m + (0 + a_col_offset): 1,
            (0 + a_row_offset) * m + (0 + a_col_offset): -1
        }
        v6 = {
            (0 + b_row_offset) * p + (0 + b_col_offset): 1,
            (0 + b_row_offset) * p + (1 + b_col_offset): 1
        }
        w6 = {
            (1 + c_col_offset) * n + (1 + c_row_offset): 1   # C11
        }
        add_term(u6, v6, w6)
        
        # P7 = (A01 - A11)(B10 + B11)
        u7 = {
            (0 + a_row_offset) * m + (1 + a_col_offset): 1,
            (1 + a_row_offset) * m + (1 + a_col_offset): -1
        }
        v7 = {
            (1 + b_row_offset) * p + (0 + b_col_offset): 1,
            (1 + b_row_offset) * p + (1 + b_col_offset): 1
        }
        w7 = {
            (0 + c_col_offset) * n + (0 + c_row_offset): 1   # C00
        }
        add_term(u7, v7, w7)

    def add_outer_product(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Adds decomposition for outer product of A block and B block.
        A block: rows in a_rows, cols in a_cols.
        B block: rows in b_rows, cols in b_cols.
        C block: rows in c_rows, cols in c_cols.
        Assumes A block is (len(a_rows) x len(a_cols)) and B block is (len(b_rows) x len(b_cols)).
        And len(a_rows) == len(c_rows), len(b_cols) == len(c_cols).
        """
        for i, ar in enumerate(a_rows):
            for j, ac in enumerate(a_cols):
                for k, br in enumerate(b_rows):
                    for l, bc in enumerate(b_cols):
                        # Term: A[ar, ac] * B[br, bc] -> C[c_rows[i], c_cols[l]]
                        # Wait, A is (rows x cols), B is (rows x cols).
                        # Product C[i, l] = sum_k A[i, k] B[k, l].
                        # Here A block indices are (ar, ac), B block indices are (br, bc).
                        # The summation index is the column of A and row of B.
                        # So ac must correspond to br.
                        # In our case, A block is 2x1 or 1x2 or 1x1.
                        # If A is 2x1, cols are [2]. B is 1x2, rows are [2].
                        # So ac=2, br=2. They match.
                        
                        u_idx = ar * m + ac
                        v_idx = br * p + bc
                        w_idx = c_cols[l] * n + c_rows[i]
                        
                        add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})

    # 1. A11 (2x2) * B11 (2x2) -> C11 (2x2) using Strassen
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # 2. A11 (2x2) * B12 (2x2) -> C12 (2x2) using Strassen
    add_strassen(0, 0, 0, 2, 0, 2)
    
    # 3. A12 (2x1) * B21 (1x2) -> C11 (2x2) Outer Product
    # A rows 0,1 col 2. B row 2 cols 0,1. C rows 0,1 cols 0,1.
    add_outer_product([0, 1], [2], [2], [0, 1], [0, 1], [0, 1])
    
    # 4. A12 (2x1) * B22 (1x2) -> C12 (2x2) Outer Product
    # A rows 0,1 col 2. B row 2 cols 2,3. C rows 0,1 cols 2,3.
    add_outer_product([0, 1], [2], [2], [2, 3], [0, 1], [2, 3])
    
    # 5. A21 (1x2) * B11 (2x2) -> C21 (1x2) Outer Product
    # A row 2 cols 0,1. B rows 0,1 cols 0,1. C row 2 cols 0,1.
    add_outer_product([2], [0, 1], [0, 1], [0, 1], [2], [0, 1])
    
    # 6. A21 (1x2) * B12 (2x2) -> C22 (1x2) Outer Product
    # A row 2 cols 0,1. B rows 0,1 cols 2,3. C row 2 cols 2,3.
    add_outer_product([2], [0, 1], [0, 1], [2, 3], [2], [2, 3])
    
    # 7. A22 (1x1) * B21 (1x2) -> C21 (1x2) Outer Product
    # A row 2 col 2. B row 2 cols 0,1. C row 2 cols 0,1.
    add_outer_product([2], [2], [2], [0, 1], [2], [0, 1])
    
    # 8. A22 (1x1) * B22 (1x2) -> C22 (1x2) Outer Product
    # A row 2 col 2. B row 2 cols 2,3. C row 2 cols 2,3.
    add_outer_product([2], [2], [2], [2, 3], [2], [2, 3])
    
    rank = len(U_cols)
    
    # Transpose to get rows
    U = [list(col) for col in zip(*U_cols)] if U_cols else []
    V = [list(col) for col in zip(*V_cols)] if V_cols else []
    W = [list(col) for col in zip(*W_cols)] if W_cols else []
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
