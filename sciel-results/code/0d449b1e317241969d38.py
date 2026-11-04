import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # We construct a rank 68 decomposition using block decomposition
    # A = [[A22, A21], [A12, A11]]
    # B = [B2x8; B1x8]
    # C = [A22*B2x8 + A21*B1x8; A12*B2x8 + A11*B1x8]
    
    # Part 1: A22 * B2x8 using Strassen (4 blocks of 2x2)
    # Part 2: A21 * B1x8 (16 mults)
    # Part 3: A12 * B2x8 (16 mults)
    # Part 4: A11 * B1x8 (8 mults)
    
    R = 68
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    # Helper to add to matrices
    def add_entry(mat, row, col_idx, val):
        mat[row, col_idx] += val

    # --- Part 1: Strassen for A22 * B2x8 ---
    # A22 indices: i in 0,1; j in 0,1
    # B2x8 indices: j in 0,1; k in 0..7
    # C indices: i in 0,1; k in 0..7
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        # U: A00, A11
        add_entry(U, 0*3 + 0, col, 1)
        add_entry(U, 1*3 + 1, col, 1)
        # V: B0k0, B1k1
        add_entry(V, 0*8 + k0, col, 1)
        add_entry(V, 1*8 + k1, col, 1)
        # W: C0k0, C1k1
        add_entry(W, k0*3 + 0, col, 1)
        add_entry(W, k1*3 + 1, col, 1)
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        add_entry(U, 1*3 + 0, col, 1)
        add_entry(U, 1*3 + 1, col, 1)
        add_entry(V, 0*8 + k0, col, 1)
        # W: C1k0
        add_entry(W, k0*3 + 1, col, 1)
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        add_entry(U, 0*3 + 0, col, 1)
        add_entry(V, 0*8 + k1, col, 1)
        add_entry(V, 1*8 + k1, col, -1)
        # W: C0k1
        add_entry(W, k1*3 + 0, col, 1)
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        add_entry(U, 1*3 + 1, col, 1)
        add_entry(V, 1*8 + k0, col, 1)
        add_entry(V, 0*8 + k0, col, -1)
        # W: C0k0, C1k0
        add_entry(W, k0*3 + 0, col, 1)
        add_entry(W, k0*3 + 1, col, 1)
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        add_entry(U, 0*3 + 0, col, 1)
        add_entry(U, 0*3 + 1, col, 1)
        add_entry(V, 1*8 + k1, col, 1)
        # W: C0k0, C0k1
        add_entry(W, k0*3 + 0, col, -1)
        add_entry(W, k1*3 + 0, col, 1)
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        add_entry(U, 1*3 + 0, col, 1)
        add_entry(U, 0*3 + 0, col, -1)
        add_entry(V, 0*8 + k0, col, 1)
        add_entry(V, 0*8 + k1, col, 1)
        # W: C1k1
        add_entry(W, k1*3 + 1, col, 1)
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        add_entry(U, 0*3 + 1, col, 1)
        add_entry(U, 1*3 + 1, col, -1)
        add_entry(V, 1*8 + k0, col, 1)
        add_entry(V, 1*8 + k1, col, 1)
        # W: C0k0, C1k1
        add_entry(W, k0*3 + 0, col, 1)
        add_entry(W, k1*3 + 1, col, 1)
        col += 1
        
    # --- Part 2: A21 * B1x8 ---
    # A21: i in 0,1; j=2
    # B1x8: j=2; k in 0..7
    # C: i in 0,1; k in 0..7
    for i in range(2):
        for k in range(8):
            # U: A_i2
            add_entry(U, i*3 + 2, col, 1)
            # V: B_2k
            add_entry(V, 2*8 + k, col, 1)
            # W: C_ik
            add_entry(W, k*3 + i, col, 1)
            col += 1
            
    # --- Part 3: A12 * B2x8 ---
    # A12: i=2; j in 0,1
    # B2x8: j in 0,1; k in 0..7
    # C: i=2; k in 0..7
    for j in range(2):
        for k in range(8):
            # U: A_2j
            add_entry(U, 2*3 + j, col, 1)
            # V: B_jk
            add_entry(V, j*8 + k, col, 1)
            # W: C_2k
            add_entry(W, k*3 + 2, col, 1)
            col += 1
            
    # --- Part 4: A11 * B1x8 ---
    # A11: i=2; j=2
    # B1x8: j=2; k in 0..7
    # C: i=2; k in 0..7
    for k in range(8):
        # U: A_22
        add_entry(U, 2*3 + 2, col, 1)
        # V: B_2k
        add_entry(V, 2*8 + k, col, 1)
        # W: C_2k
        add_entry(W, k*3 + 2, col, 1)
        col += 1
        
    # Convert to list of lists
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
