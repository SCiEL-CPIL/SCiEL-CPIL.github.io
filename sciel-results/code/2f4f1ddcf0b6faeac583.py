import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 5 Strassen blocks for the first 10 columns and 1 naive block for the last column.
    # Strassen rank for 2x2x2 is 7.
    # Naive rank for 2x2x1 is 4.
    # Total rank = 5 * 7 + 4 = 39.
    rank = 39
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term to U, V, W
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in enumerate(u_vals):
            if val != 0:
                U[idx, col] = val
        for idx, val in enumerate(v_vals):
            if val != 0:
                V[idx, col] = val
        for idx, val in enumerate(w_vals):
            if val != 0:
                W[idx, col] = val
        col += 1

    # Strassen terms for 2x2x2 block
    # A indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # B indices: (0,0), (0,1), (1,0), (1,1) relative to block
    # C indices: (0,0), (0,1), (1,0), (1,1) relative to block
    
    def strassen_block(col_start, k_offset):
        # k_offset is the starting column index for B and C
        # U indices are global (i*m + j)
        # V indices are global (j*p + k)
        # W indices are global (k*n + i)
        
        # M1 = (A11 + A22)(B11 + B22)
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 1, 0] # B11, B12, B21, B22? No, V is (j, k).
        # B11 is j=0, k=k_offset. B22 is j=1, k=k_offset+1.
        # V indices: j*p + k.
        v1 = [0]* (m*p)
        v1[0*p + k_offset] = 1
        v1[1*p + k_offset + 1] = 1
        w1 = [0]* (p*n)
        # C11 = M1 + ... -> W[0*n+0] += 1
        # C22 = M1 + ... -> W[1*n+1] += 1
        w1[0*n + 0] = 1
        w1[1*n + 1] = 1
        add_term(u1, v1, w1)
        
        # M2 = (A21 + A22)B11
        u2 = [0, 0, 1, 1]
        v2 = [0]* (m*p)
        v2[0*p + k_offset] = 1
        w2 = [0]* (p*n)
        # C21 = M2 + ... -> W[1*n+0] += 1
        # C22 = M1 - M2 + ... -> W[1*n+1] -= 1
        w2[1*n + 0] = 1
        w2[1*n + 1] = -1
        add_term(u2, v2, w2)
        
        # M3 = A11(B12 - B22)
        u3 = [1, 0, 0, 0]
        v3 = [0]* (m*p)
        v3[0*p + k_offset + 1] = 1
        v3[1*p + k_offset + 1] = -1
        w3 = [0]* (p*n)
        # C12 = M3 + ... -> W[0*n+1] += 1
        # C22 = M1 - M2 + M3 + ... -> W[1*n+1] += 1
        w3[0*n + 1] = 1
        w3[1*n + 1] = 1
        add_term(u3, v3, w3)
        
        # M4 = A22(B21 - B11)
        u4 = [0, 0, 0, 1]
        v4 = [0]* (m*p)
        v4[1*p + k_offset] = 1
        v4[0*p + k_offset] = -1
        w4 = [0]* (p*n)
        # C11 = M1 + M4 - ... -> W[0*n+0] += 1
        # C21 = M2 + M4 -> W[1*n+0] += 1
        w4[0*n + 0] = 1
        w4[1*n + 0] = 1
        add_term(u4, v4, w4)
        
        # M5 = (A11 + A12)B22
        u5 = [1, 1, 0, 0]
        v5 = [0]* (m*p)
        v5[1*p + k_offset + 1] = 1
        w5 = [0]* (p*n)
        # C11 = M1 + M4 - M5 + ... -> W[0*n+0] -= 1
        # C12 = M3 + M5 -> W[0*n+1] += 1
        w5[0*n + 0] = -1
        w5[0*n + 1] = 1
        add_term(u5, v5, w5)
        
        # M6 = (A21 - A11)(B11 + B12)
        u6 = [0, 0, 1, -1]
        v6 = [0]* (m*p)
        v6[0*p + k_offset] = 1
        v6[0*p + k_offset + 1] = 1
        w6 = [0]* (p*n)
        # C22 = ... + M6 -> W[1*n+1] += 1
        w6[1*n + 1] = 1
        add_term(u6, v6, w6)
        
        # M7 = (A12 - A22)(B21 + B22)
        u7 = [0, 1, 0, -1]
        v7 = [0]* (m*p)
        v7[1*p + k_offset] = 1
        v7[1*p + k_offset + 1] = 1
        w7 = [0]* (p*n)
        # C11 = ... + M7 -> W[0*n+0] += 1
        w7[0*n + 0] = 1
        add_term(u7, v7, w7)

    # Naive block for 2x2x1
    def naive_block(k_offset):
        # M1 = A11 B11
        u1 = [1, 0, 0, 0]
        v1 = [0]* (m*p)
        v1[0*p + k_offset] = 1
        w1 = [0]* (p*n)
        w1[0*n + 0] = 1 # C11
        add_term(u1, v1, w1)
        
        # M2 = A12 B21
        u2 = [0, 1, 0, 0]
        v2 = [0]* (m*p)
        v2[1*p + k_offset] = 1
        w2 = [0]* (p*n)
        w2[0*n + 0] = 1 # C11
        add_term(u2, v2, w2)
        
        # M3 = A21 B11
        u3 = [0, 0, 1, 0]
        v3 = [0]* (m*p)
        v3[0*p + k_offset] = 1
        w3 = [0]* (p*n)
        w3[1*n + 0] = 1 # C21
        add_term(u3, v3, w3)
        
        # M4 = A22 B21
        u4 = [0, 0, 0, 1]
        v4 = [0]* (m*p)
        v4[1*p + k_offset] = 1
        w4 = [0]* (p*n)
        w4[1*n + 0] = 1 # C21
        add_term(u4, v4, w4)

    # Apply 5 Strassen blocks for k=0, 2, 4, 6, 8
    for t in range(5):
        strassen_block(t, 2*t)
        
    # Apply 1 Naive block for k=10
    naive_block(10)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
