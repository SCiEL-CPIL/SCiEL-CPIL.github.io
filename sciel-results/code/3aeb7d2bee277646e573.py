import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) -> (8, rank)
    # V shape: (m*p, rank) -> (16, rank)
    # W shape: (p*n, rank) -> (8, rank)
    
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen coefficients for 2x2 multiplication
    # P1 = (X11+X22)(Y11+Y22)
    # P2 = (X21+X22)Y11
    # P3 = X11(Y12-Y22)
    # P4 = X22(Y21-Y11)
    # P5 = (X11+X12)Y22
    # P6 = (X21-X11)(Y11+Y12)
    # P7 = (X12-X22)(Y21+Y22)
    
    # Output reconstruction:
    # Z11 = P1 + P4 - P5 + P7
    # Z12 = P3 + P5
    # Z21 = P2 + P4
    # Z22 = P1 - P2 + P3 + P6
    
    # Indices in 2x2 block (row, col)
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    # Define Strassen terms: (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    # Note: W_indices are relative to the 2x2 output block.
    
    strassen_terms = [
        # P1
        ([(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1], [(0,0), (1,1)], [1, 1]),
        # P2
        ([(1,0), (1,1)], [1, 1], [(0,0)], [1], [(1,0), (1,1)], [1, -1]), # Z21 gets +P2, Z22 gets -P2
        # P3
        ([(0,0)], [1], [(0,1), (1,1)], [1, -1], [(0,1), (1,1)], [1, 1]),   # Z12 gets +P3, Z22 gets +P3
        # P4
        ([(1,1)], [1], [(1,0), (0,0)], [1, -1], [(0,0), (1,0)], [1, 1]),    # Z11 gets +P4, Z21 gets +P4
        # P5
        ([(0,0), (0,1)], [1, 1], [(1,1)], [1], [(0,0), (0,1)], [-1, 1]),    # Z11 gets -P5, Z12 gets +P5
        # P6
        ([(1,0), (0,0)], [1, -1], [(0,0), (0,1)], [1, 1], [(1,1)], [1]),    # Z22 gets +P6
        # P7
        ([(0,1), (1,1)], [1, -1], [(1,0), (1,1)], [1, 1], [(0,0), (0,1)], [1, 1]), # Z11 gets +P7, Z12 gets +P7
    ]
    
    # Map global indices
    # A: 2x4. A1 is cols 0-1, A2 is cols 2-3.
    # B: 4x4. B11 is rows 0-1, cols 0-1. B12 is rows 0-1, cols 2-3.
    #      B21 is rows 2-3, cols 0-1. B22 is rows 2-3, cols 2-3.
    # C: 2x4. C1 is cols 0-1, C2 is cols 2-3.
    
    # Products to compute:
    # 1. A1 * B11 -> C1
    # 2. A2 * B21 -> C1
    # 3. A1 * B12 -> C2
    # 4. A2 * B22 -> C2
    
    # For each product, we add 7 terms.
    
    def add_term(prod_idx, term_idx, a_block, b_block, c_block, s_term):
        u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff = s_term
        col = prod_idx * 7 + term_idx
        
        # A indices
        # a_block: 0 for A1, 1 for A2
        # A1 is A[:, 0:2], A2 is A[:, 2:4]
        # Local (r, c) in 2x2 block -> Global (r, 2*a_block + c)
        for loc_idx, coeff in zip(u_idx, u_coeff):
            r, c = loc_idx
            global_r = r
            global_c = 2 * a_block + c
            U[global_r * m + global_c, col] += coeff
            
        # B indices
        # b_block: 0 for B11, 1 for B12, 2 for B21, 3 for B22
        # B11: rows 0-1, cols 0-1
        # B12: rows 0-1, cols 2-3
        # B21: rows 2-3, cols 0-1
        # B22: rows 2-3, cols 2-3
        # Local (r, c) in 2x2 block
        # Row offset: 2 * (b_block // 2)
        # Col offset: 2 * (b_block % 2)
        row_off = 2 * (b_block // 2)
        col_off = 2 * (b_block % 2)
        for loc_idx, coeff in zip(v_idx, v_coeff):
            r, c = loc_idx
            global_r = row_off + r
            global_c = col_off + c
            V[global_r * p + global_c, col] += coeff
            
        # C indices
        # c_block: 0 for C1, 1 for C2
        # C1 is C[:, 0:2], C2 is C[:, 2:4]
        # Local (r, c) in 2x2 block
        # Global (r, 2*c_block + c)
        for loc_idx, coeff in zip(w_idx, w_coeff):
            r, c = loc_idx
            global_r = r
            global_c = 2 * c_block + c
            W[global_c * n + global_r, col] += coeff # Trace convention W[k*n+i, r] -> W[col*n + row]

    # Product 0: A1 * B11 -> C1
    for t in range(7):
        add_term(0, t, 0, 0, 0, strassen_terms[t])
        
    # Product 1: A2 * B21 -> C1
    for t in range(7):
        add_term(1, t, 1, 2, 0, strassen_terms[t])
        
    # Product 2: A1 * B12 -> C2
    for t in range(7):
        add_term(2, t, 0, 1, 1, strassen_terms[t])
        
    # Product 3: A2 * B22 -> C2
    for t in range(7):
        add_term(3, t, 1, 3, 1, strassen_terms[t])
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
