import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 49 Strassen decomposition
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen steps for 2x2 matrices
    # Each step s defines:
    # x_part: coefficients for x_00, x_10, x_01, x_11
    # y_part: coefficients for y_00, y_10, y_01, y_11
    # Note: x is X matrix, y is Y matrix.
    # Indices: 00, 10, 01, 11 -> (row, col)
    
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)y00
    # m3 = x00(y01-y11)
    # m4 = x11(y10-y00)
    # m5 = (x00+x01)y11
    # m6 = (x10-x00)(y00+y01)
    # m7 = (x01-x11)(y10+y11)
    
    # Coefficients for X part of each m_s (order: x00, x10, x01, x11)
    # Coefficients for Y part of each m_s (order: y00, y10, y01, y11)
    
    x_coeffs = [
        [1, 0, 0, 1], # m1
        [0, 1, 0, 1], # m2
        [1, 0, 0, 0], # m3
        [0, 0, 0, 1], # m4
        [1, 0, 1, 0], # m5
        [-1, 1, 0, 0],# m6
        [0, 0, 1, -1] # m7
    ]
    
    y_coeffs = [
        [1, 0, 0, 1], # m1
        [1, 0, 0, 0], # m2
        [0, 0, 1, -1],# m3
        [0, 0, -1, 1],# m4
        [0, 0, 0, 1], # m5
        [1, 0, 1, 0], # m6
        [0, 1, 0, 1]  # m7
    ]
    
    # Coefficients for result P_uv in terms of m_s
    # P00 = m1 + m4 - m5 + m7
    # P01 = m3 + m5
    # P10 = m2 + m4
    # P11 = m1 - m2 + m3 + m6
    # Order: m1, m2, m3, m4, m5, m6, m7
    p_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # P00
        [0, 0, 1, 0, 1, 0, 0],  # P01
        [0, 1, 0, 1, 0, 0, 0],  # P10
        [1, -1, 1, 0, 0, 1, 0] # P11
    ]
    
    # Define the 7 groups (M1..M7)
    # Each group g defines:
    # X blocks: which 2x2 blocks of A are combined to form X
    # Y blocks: which 2x2 blocks of B are combined to form Y
    # Signs: how P (result of X*Y) is added to C blocks
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # X definition: list of (block, coeff)
    # Block indices: 00, 10, 01, 11 -> (row_block, col_block)
    # A blocks:
    # A00: i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    # Y definition: list of (block, coeff)
    # B blocks:
    # B00: j in 0,1; k in 0,1
    # B01: j in 0,1; k in 2,3
    # B10: j in 2,3; k in 0,1
    # B11: j in 2,3; k in 2,3
    
    groups = [
        { # M1
            'X': [(0,0,1), (1,1,1)],
            'Y': [(0,0,1), (1,1,1)],
            'C_signs': {(0,0): 1, (1,1): 1}
        },
        { # M2
            'X': [(1,0,1), (1,1,1)],
            'Y': [(0,0,1)],
            'C_signs': {(1,0): 1, (1,1): -1}
        },
        { # M3
            'X': [(0,0,1)],
            'Y': [(0,1,1), (1,1,-1)],
            'C_signs': {(0,1): 1, (1,1): 1}
        },
        { # M4
            'X': [(1,1,1)],
            'Y': [(1,0,1), (0,0,-1)],
            'C_signs': {(0,0): 1, (1,0): 1}
        },
        { # M5
            'X': [(0,0,1), (0,1,1)],
            'Y': [(1,1,1)],
            'C_signs': {(0,0): -1, (0,1): 1}
        },
        { # M6
            'X': [(1,0,1), (0,0,-1)],
            'Y': [(0,0,1), (0,1,1)],
            'C_signs': {(1,1): 1}
        },
        { # M7
            'X': [(0,1,1), (1,1,-1)],
            'Y': [(1,0,1), (1,1,1)],
            'C_signs': {(0,0): 1}
        }
    ]
    
    for g_idx, group in enumerate(groups):
        # For each of the 7 sub-steps s
        for s_idx in range(7):
            col = g_idx * 7 + s_idx
            
            # Get coefficients for X and Y parts of this sub-step
            x_c = x_coeffs[s_idx]
            y_c = y_coeffs[s_idx]
            
            # Build U and V
            # U[i*m+j, col]
            # V[j*p+k, col]
            
            # Iterate over all i, j
            for i in range(4):
                for j in range(4):
                    # Determine which A block (rb, cb) this (i,j) belongs to
                    rb = i // 2
                    cb = j // 2
                    # Sub-indices within block
                    sr = i % 2
                    sc = j % 2
                    
                    # Check if this block is in group['X']
                    coeff = 0
                    for (br, bc, c) in group['X']:
                        if br == rb and bc == cb:
                            # The sub-index mapping for X:
                            # X is 2x2. X_00 corresponds to A_00's (0,0) etc.
                            # So X_sr_sc is the entry.
                            # x_c index: 00, 10, 01, 11 -> (0, 1, 2, 3)
                            # My x_c list order: x00, x10, x01, x11
                            # Index 0: (0,0)
                            # Index 1: (1,0)
                            # Index 2: (0,1)
                            # Index 3: (1,1)
                            idx = sr * 2 + sc # 0,1,2,3
                            # Wait, my x_c order is x00, x10, x01, x11.
                            # idx 0: 0,0 -> x00
                            # idx 1: 1,0 -> x10
                            # idx 2: 0,1 -> x01
                            # idx 3: 1,1 -> x11
                            coeff += c * x_c[idx]
                            break
                    
                    U[i * 4 + j][col] = coeff
            
            for j in range(4):
                for k in range(4):
                    # Determine which B block (rb, cb) this (j,k) belongs to
                    rb = j // 2
                    cb = k // 2
                    sr = j % 2
                    sc = k % 2
                    
                    coeff = 0
                    for (br, bc, c) in group['Y']:
                        if br == rb and bc == cb:
                            idx = sr * 2 + sc
                            coeff += c * y_c[idx]
                            break
                    
                    V[j * 4 + k][col] = coeff
            
            # Build W
            # W[k*n+i, col]
            # C_signs maps (block_row, block_col) to sign
            # P_coeffs[u, v, s] is coeff of m_s in P_uv
            
            for i in range(4):
                for k in range(4):
                    # Determine which C block (rb, cb) this (i,k) belongs to
                    rb = i // 2
                    cb = k // 2
                    
                    sign = group['C_signs'].get((rb, cb), 0)
                    if sign != 0:
                        # Get coefficient of m_s in P_rb_cb
                        # Map (rb, cb) to flat index for p_coeffs
                        block_idx = rb * 2 + cb
                        coeff = sign * p_coeffs[block_idx][s_idx]
                        W[k * 4 + i][col] = coeff
                    else:
                        W[k * 4 + i][col] = 0
                        
    # Time guard check
    if time.time() > deadline:
        # Return whatever we have (should be complete)
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
