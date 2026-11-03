import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # We construct a decomposition of rank 72.
    # Part 1: Rank 56 decomposition for the first 4 columns (M_4,4,4) using block Strassen.
    # Part 2: Rank 16 schoolbook decomposition for the 5th column (M_4,4,1).
    
    rank_total = 72
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal rank_current
        # u_idx: list of (row, col) in A
        # v_idx: list of (row, col) in B
        # w_idx: list of (row, col) in C
        # This is not quite right for Strassen, which involves sums.
        # Strassen terms are rank-1 in the tensor sense?
        # No, Strassen reduces the number of multiplications.
        # The tensor decomposition terms are the intermediate products.
        # Each intermediate product is a rank-1 tensor in the subspace?
        # Actually, the Strassen algorithm provides a decomposition into 7 rank-1 tensors.
        
        pass

    # Let's explicitly define the 7 Strassen terms for 2x2x2.
    # Indices: A(0,0)=0, A(0,1)=1, A(1,0)=2, A(1,1)=3
    # B(0,0)=0, B(0,1)=1, B(1,0)=2, B(1,1)=3
    # C(0,0)=0, C(0,1)=1, C(1,0)=2, C(1,1)=3
    
    # Term 1: (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1
    # Term 2: (A10+A11)B00
    # U: A10=1, A11=1
    # V: B00=1
    # W: C10=1, C11=-1
    # Term 3: A00(B01-B11)
    # U: A00=1
    # V: B01=1, B11=-1
    # W: C01=1, C11=1
    # Term 4: A11(B10-B00)
    # U: A11=1
    # V: B10=1, B00=-1
    # W: C10=1, C11=1
    # Term 5: (A00+A01)B11
    # U: A00=1, A01=1
    # V: B11=1
    # W: C11=-1, C01=1
    # Term 6: (A10-A00)(B00+B01)
    # U: A10=1, A00=-1
    # V: B00=1, B01=1
    # W: C11=1
    # Term 7: (A01-A11)(B10+B11)
    # U: A01=1, A11=-1
    # V: B10=1, B11=1
    # W: C00=1

    strassen_terms = [
        # (U_offsets, V_offsets, W_offsets)
        # Offsets are relative to the block start.
        # A: (r, c) -> r*2+c
        # B: (r, c) -> r*2+c
        # C: (r, c) -> r*2+c
        [
            {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},
        ],
        [
            {2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1},
        ],
        [
            {0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1},
        ],
        [
            {3: 1}, {2: 1, 0: -1}, {2: 1, 3: 1},
        ],
        [
            {0: 1, 1: 1}, {3: 1}, {3: -1, 1: 1},
        ],
        [
            {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1},
        ],
        [
            {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1},
        ],
    ]

    rank_current = 0
    
    # Apply Strassen to the 8 blocks of M_4,4,4
    for i_block in range(2):
        for j_block in range(2):
            # A block i_block, 0 and i_block, 1
            # B block 0, j_block and 1, j_block
            # C block i_block, j_block
            
            # Term 1: A[i_block, 0] * B[0, j_block]
            # This is a 2x2x2 multiplication.
            # A rows: 2*i_block, 2*i_block+1
            # A cols: 0, 1
            # B rows: 0, 1
            # B cols: 2*j_block, 2*j_block+1
            # C rows: 2*i_block, 2*i_block+1
            # C cols: 2*j_block, 2*j_block+1
            
            for t in range(7):
                U_off, V_off, W_off = strassen_terms[t]
                
                # Map U offsets
                for u_idx, val in U_off.items():
                    r_off, c_off = divmod(u_idx, 2)
                    a_row = 2 * i_block + r_off
                    a_col = c_off # Block 0 cols are 0,1
                    U[a_row * m + a_col, rank_current] += val
                
                # Map V offsets
                for v_idx, val in V_off.items():
                    r_off, c_off = divmod(v_idx, 2)
                    b_row = r_off # Block 0 rows are 0,1
                    b_col = 2 * j_block + c_off
                    V[b_row * p + b_col, rank_current] += val
                
                # Map W offsets
                for w_idx, val in W_off.items():
                    r_off, c_off = divmod(w_idx, 2)
                    c_row = 2 * i_block + r_off
                    c_col = 2 * j_block + c_off
                    W[c_col * n + c_row, rank_current] += val
                
                rank_current += 1
            
            # Term 2: A[i_block, 1] * B[1, j_block]
            # A rows: 2*i_block, 2*i_block+1
            # A cols: 2, 3
            # B rows: 2, 3
            # B cols: 2*j_block, 2*j_block+1
            # C rows: 2*i_block, 2*i_block+1
            # C cols: 2*j_block, 2*j_block+1
            
            for t in range(7):
                U_off, V_off, W_off = strassen_terms[t]
                
                # Map U offsets
                for u_idx, val in U_off.items():
                    r_off, c_off = divmod(u_idx, 2)
                    a_row = 2 * i_block + r_off
                    a_col = 2 + c_off # Block 1 cols are 2,3
                    U[a_row * m + a_col, rank_current] += val
                
                # Map V offsets
                for v_idx, val in V_off.items():
                    r_off, c_off = divmod(v_idx, 2)
                    b_row = 2 + r_off # Block 1 rows are 2,3
                    b_col = 2 * j_block + c_off
                    V[b_row * p + b_col, rank_current] += val
                
                # Map W offsets
                for w_idx, val in W_off.items():
                    r_off, c_off = divmod(w_idx, 2)
                    c_row = 2 * i_block + r_off
                    c_col = 2 * j_block + c_off
                    W[c_col * n + c_row, rank_current] += val
                
                rank_current += 1

    # Part 2: Schoolbook for the 5th column (k=4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for i in range(n):
        for j in range(m):
            U[i * m + j, rank_current] = 1
            V[j * p + 4, rank_current] = 1
            W[4 * n + i, rank_current] = 1
            rank_current += 1

    return {"U": U, "V": V, "W": W}
