import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 construction
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # a_start, a_end : row indices in A (0..1)
    # j_start : col index in A (0..1) for the 2x2 block
    # b_row_start : row indices in B (0..1)
    # b_col_start : col index in B (0..3) for the 2x2 block
    # w_col_start : col index in C (0..3) for the 2x2 block
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_cols):
        nonlocal col
        # A indices: i*m + j
        a_idx = [ (a_rows[0]*m + a_cols[0]), (a_rows[0]*m + a_cols[1]), 
                  (a_rows[1]*m + a_cols[0]), (a_rows[1]*m + a_cols[1]) ]
        # A00, A01, A10, A11
        A00, A01, A10, A11 = a_idx
        
        # B indices: j*p + k
        b_idx = [ (b_rows[0]*p + b_cols[0]), (b_rows[0]*p + b_cols[1]),
                  (b_rows[1]*p + b_cols[0]), (b_rows[1]*p + b_cols[1]) ]
        # B00, B01, B10, B11
        B00, B01, B10, B11 = b_idx
        
        # C indices: k*n + i
        # c_cols are the k indices (0..1 relative to block start)
        # We need global k indices
        # The function call passes global indices for b_cols, so we can derive k indices?
        # No, b_cols are global column indices of B.
        # C column index corresponds to B column index.
        # C row index corresponds to A row index.
        
        # C00: i=a_rows[0], k=b_cols[0]
        C00 = b_cols[0]*n + a_rows[0]
        # C01: i=a_rows[0], k=b_cols[1]
        C01 = b_cols[1]*n + a_rows[0]
        # C10: i=a_rows[1], k=b_cols[0]
        C10 = b_cols[0]*n + a_rows[1]
        # C11: i=a_rows[1], k=b_cols[1]
        C11 = b_cols[1]*n + a_rows[1]
        
        # Factor 1: (A00+A11)(B00+B11)
        U[A00, col] += 1
        U[A11, col] += 1
        V[B00, col] += 1
        V[B11, col] += 1
        W[C00, col] += 1
        W[C11, col] += 1
        col += 1
        
        # Factor 2: (A10+A11)B00
        U[A10, col] += 1
        U[A11, col] += 1
        V[B00, col] += 1
        W[C10, col] += 1
        W[C11, col] += -1
        col += 1
        
        # Factor 3: A00(B01-B11)
        U[A00, col] += 1
        V[B01, col] += 1
        V[B11, col] += -1
        W[C01, col] += 1
        W[C11, col] += 1
        col += 1
        
        # Factor 4: A11(B10-B00)
        U[A11, col] += 1
        V[B10, col] += 1
        V[B00, col] += -1
        W[C00, col] += 1
        W[C10, col] += 1
        col += 1
        
        # Factor 5: (A00+A01)B11
        U[A00, col] += 1
        U[A01, col] += 1
        V[B11, col] += 1
        W[C00, col] += -1
        W[C01, col] += 1
        col += 1
        
        # Factor 6: (A10-A00)(B00+B01)
        U[A10, col] += 1
        U[A00, col] += -1
        V[B00, col] += 1
        V[B01, col] += 1
        W[C11, col] += 1
        col += 1
        
        # Factor 7: (A01-A11)(B10+B11)
        U[A01, col] += 1
        U[A11, col] += -1
        V[B10, col] += 1
        V[B11, col] += 1
        W[C00, col] += 1
        W[C11, col] += -1
        col += 1
        
    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_2x2x1(a_rows, a_cols, b_rows, b_col, c_col):
        nonlocal col
        # A indices
        A00 = a_rows[0]*m + a_cols[0]
        A01 = a_rows[0]*m + a_cols[1]
        A10 = a_rows[1]*m + a_cols[0]
        A11 = a_rows[1]*m + a_cols[1]
        
        # B indices
        B0 = b_rows[0]*p + b_col
        B1 = b_rows[1]*p + b_col
        
        # C indices
        C0 = c_col*n + a_rows[0]
        C1 = c_col*n + a_rows[1]
        
        # Factor 1: A00*B0
        U[A00, col] += 1
        V[B0, col] += 1
        W[C0, col] += 1
        col += 1
        
        # Factor 2: A01*B1
        U[A01, col] += 1
        V[B1, col] += 1
        W[C0, col] += 1
        col += 1
        
        # Factor 3: A10*B0
        U[A10, col] += 1
        V[B0, col] += 1
        W[C1, col] += 1
        col += 1
        
        # Factor 4: A11*B1
        U[A11, col] += 1
        V[B1, col] += 1
        W[C1, col] += 1
        col += 1

    # Helper to add Schoolbook 2x1x5 block
    def add_schoolbook_2x1x5(a_rows, a_col, b_row, b_cols, c_cols):
        nonlocal col
        for i in a_rows:
            for k in b_cols:
                U[i*m + a_col, col] += 1
                V[b_row*p + k, col] += 1
                W[k*n + i, col] += 1
                col += 1

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A rows 0,1. A cols 0,1.
    # B rows 0,1. B cols 0,1.
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Part 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    # A rows 0,1. A cols 0,1.
    # B rows 0,1. B cols 2,3.
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], [2, 3])
    
    # Part 3: Schoolbook 2x2x1 for A[0:2, 0:2] * B[0:2, 4:5]
    # A rows 0,1. A cols 0,1.
    # B rows 0,1. B col 4.
    add_schoolbook_2x2x1([0, 1], [0, 1], [0, 1], 4, 4)
    
    # Part 4: Schoolbook 2x1x5 for A[0:2, 2:3] * B[2:3, 0:5]
    # A rows 0,1. A col 2.
    # B row 2. B cols 0,1,2,3,4.
    add_schoolbook_2x1x5([0, 1], 2, 2, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
