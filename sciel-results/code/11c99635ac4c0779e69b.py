import numpy as np

def construct():
    n, m, p = 3, 4, 4
    # Strategy: Block-diagonal decomposition into 1x4x4 and 2x4x4
    # Rank 1x4x4 = 16
    # Rank 2x4x4 = 32
    # Total Rank = 48
    
    rank = 48
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Part 1: 1x4x4 (Row 0 of A, Row 0 of C) ---
    # A indices: 0..3 (col 0..3 of row 0)
    # B indices: 0..15
    # C indices: 0..3 (col 0..3 of row 0)
    # C_j = sum_{i=0}^3 A_i B_{i,j}
    # Rank 16 decomposition:
    # For each i (0..3) and j (0..3):
    #   Term k = i*4 + j
    #   U[A_idx, k] = 1
    #   V[B_idx, k] = 1
    #   W[C_idx, k] = 1
    
    # A is 3x4. Row 0 is A[0, 0..3]. Global A indices: 0, 1, 2, 3.
    # B is 4x4. Global B indices: r*4 + c.
    # C is 3x4. Row 0 is C[0, 0..3]. Global C indices: c*3 + 0? 
    # Convention: W[k*n+i, r] where k is col, i is row?
    # Task says: W[k*n+i, r]. 
    # In current program: global_c_idx = c_col_start * n + c_row_start.
    # So C index is col * 3 + row.
    
    # Part 1: Row 0 of A (indices 0-3 in A), Row 0 of C (indices 0, 3, 6, 9 in C? No)
    # C[0, j] index = j * 3 + 0 = 0, 3, 6, 9.
    
    for i in range(4):
        for j in range(4):
            k = i * 4 + j
            # A element (0, i) -> global index 0*4 + i = i
            U[i, k] = 1
            # B element (i, j) -> global index i*4 + j
            V[i * 4 + j, k] = 1
            # C element (0, j) -> global index j*3 + 0 = 3*j
            W[3 * j, k] = 1
            
    # --- Part 2: 2x4x4 (Rows 1,2 of A, Rows 1,2 of C) ---
    # A indices: Row 1 (4..7), Row 2 (8..11)
    # B indices: 0..15
    # C indices: Row 1 (1, 4, 7, 10), Row 2 (2, 5, 8, 11)
    # Rank 32: Two independent 1x4x4 multiplications.
    
    offset = 16
    
    # Sub-part 2a: Row 1 of A
    for i in range(4):
        for j in range(4):
            k = offset + i * 4 + j
            # A element (1, i) -> global index 1*4 + i = 4 + i
            U[4 + i, k] = 1
            # B element (i, j) -> global index i*4 + j
            V[i * 4 + j, k] = 1
            # C element (1, j) -> global index j*3 + 1 = 3*j + 1
            W[3 * j + 1, k] = 1
            
    # Sub-part 2b: Row 2 of A
    offset2 = 32
    for i in range(4):
        for j in range(4):
            k = offset2 + i * 4 + j
            # A element (2, i) -> global index 2*4 + i = 8 + i
            U[8 + i, k] = 1
            # B element (i, j) -> global index i*4 + j
            V[i * 4 + j, k] = 1
            # C element (2, j) -> global index j*3 + 2 = 3*j + 2
            W[3 * j + 2, k] = 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
