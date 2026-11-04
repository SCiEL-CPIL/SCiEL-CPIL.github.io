import numpy as np

def construct():
    n, m, p = 4, 4, 9
    rank = 0
    U_list = []
    V_list = []
    W_list = []

    # Helper to add a decomposition block
    def add_decomposition(u_vec, v_vec, w_vec):
        nonlocal rank
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        rank += 1

    # Strassen 2x2 decomposition definition
    # u, v, w are vectors of length 4 (indices 0,1,2,3 for 2x2 matrices)
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]

    # Outer Strassen definitions for 4x4
    # X_k and Y_k are linear combinations of 2x2 blocks
    # Blocks indexed 0,1,2,3
    # A blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # We need to map the "block" indices to "scalar" indices for U, V, W
    # A scalar index for block b, local index l is b*4 + l
    # B scalar index for block b, local index l is b*4 + l (for the 4x4 part)
    # C scalar index for block b, local index l is b*4 + l (for the 4x4 part)
    
    # Outer coefficients for X_k (A) and Y_k (B)
    # Each is a list of 4 coefficients for the 4 blocks
    outer_X = [
        [1, 0, 0, 1], # X1: A00 + A11
        [0, 0, 1, 1], # X2: A10 + A11
        [1, 0, 0, 0], # X3: A00
        [0, 0, 0, 1], # X4: A11
        [1, 1, 0, 0], # X5: A00 + A01
        [0, -1, 1, 0],# X6: A10 - A00 (Note: A10 is block 2, A00 is block 0)
        [0, 1, 0, -1] # X7: A01 - A11
    ]
    
    outer_Y = [
        [1, 0, 0, 1], # Y1: B00 + B11
        [1, 0, 0, 0], # Y2: B00
        [0, 1, 0, -1],# Y3: B01 - B11
        [0, 0, 1, -1],# Y4: B10 - B00
        [0, 0, 0, 1], # Y5: B11
        [1, 1, 0, 0], # Y6: B00 + B01
        [0, 0, 1, 1]  # Y7: B10 + B11
    ]
    
    outer_W = [
        [1, 0, 0, 1], # W1: C00 + C11
        [0, 0, 1, -1],# W2: C10 - C11
        [0, 1, 0, 1], # W3: C01 + C11
        [1, 0, 1, 0], # W4: C00 + C10
        [-1, 1, 0, 0],# W5: -C00 + C01
        [0, 0, 0, 1], # W6: C11
        [1, 0, 0, 0]  # W7: C00
    ]

    def generate_4x4_block(k_offset, b_offset):
        # k_offset: offset for C index (0 or 4)
        # b_offset: offset for B index (0 or 4)
        # A index offset is always 0
        
        for k in range(7):
            x_coeff = outer_X[k]
            y_coeff = outer_Y[k]
            w_coeff = outer_W[k]
            
            for s in range(7):
                u_s, v_s, w_s = strassen_2x2[s]
                
                # Construct U vector (length 16)
                u_vec = [0] * 16
                for block_idx in range(4):
                    if x_coeff[block_idx] != 0:
                        for local_idx in range(4):
                            if u_s[local_idx] != 0:
                                a_idx = block_idx * 4 + local_idx
                                u_vec[a_idx] += x_coeff[block_idx] * u_s[local_idx]
                                
                # Construct V vector (length 36)
                v_vec = [0] * 36
                for block_idx in range(4):
                    if y_coeff[block_idx] != 0:
                        for local_idx in range(4):
                            if v_s[local_idx] != 0:
                                # B index: block_idx * 4 + local_idx + b_offset (for the 4x4 part of B)
                                # Wait, B is 4x9.
                                # Blocks are 2x2.
                                # Block 0: rows 0-1, cols 0-1 (relative to offset)
                                # Block 1: rows 0-1, cols 2-3
                                # Block 2: rows 2-3, cols 0-1
                                # Block 3: rows 2-3, cols 2-3
                                # Local index l: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                                # Row = block_row * 2 + local_row
                                # Col = block_col * 2 + local_col + b_offset
                                # Block idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                                br = block_idx // 2
                                bc = block_idx % 2
                                lr = local_idx // 2
                                lc = local_idx % 2
                                row = br * 2 + lr
                                col = bc * 2 + lc + b_offset
                                b_idx = row * 9 + col
                                v_vec[b_idx] += y_coeff[block_idx] * v_s[local_idx]
                                
                # Construct W vector (length 36)
                w_vec = [0] * 36
                for block_idx in range(4):
                    if w_coeff[block_idx] != 0:
                        for local_idx in range(4):
                            if w_s[local_idx] != 0:
                                # C index: similar to B but with k_offset
                                br = block_idx // 2
                                bc = block_idx % 2
                                lr = local_idx // 2
                                lc = local_idx % 2
                                row = br * 2 + lr
                                col = bc * 2 + lc + k_offset
                                c_idx = col * 4 + row # Trace convention W[k*n+i] -> index k*4+i
                                w_vec[c_idx] += w_coeff[block_idx] * w_s[local_idx]
                                
                add_decomposition(u_vec, v_vec, w_vec)

    # Part 1: k in 0..3, B cols 0..3
    generate_4x4_block(0, 0)
    
    # Part 2: k in 4..7, B cols 4..7
    generate_4x4_block(4, 4)
    
    # Part 3: k=8, B col 8
    # Rank 16
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            
            a_idx = i * 4 + j
            b_idx = j * 9 + 8
            c_idx = 8 * 4 + i # k=8, i=i
            
            u_vec[a_idx] = 1
            v_vec[b_idx] = 1
            w_vec[c_idx] = 1
            
            add_decomposition(u_vec, v_vec, w_vec)

    U = [U_list[r] for r in range(rank)]
    V = [V_list[r] for r in range(rank)]
    W = [W_list[r] for r in range(rank)]
    
    return {"U": U, "V": V, "W": W}
