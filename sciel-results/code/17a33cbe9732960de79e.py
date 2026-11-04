import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Rank 98 decomposition: 2 blocks of 4x4x4 Strassen (Rank 49 each)
    # Total Rank = 98
    rank = 98
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 Base Coefficients
    # U2, V2, W2 are 4x7 matrices for the 2x2x2 core
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Using explicit lists for AST difference
    u2_rows = [
        [1, 0, 1, 0, 1, -1, 0], # S0, S1, S2, S3, S4, S5, S6 contributions to x00
        [0, 0, 0, 0, 1, 0, 1],  # x01
        [0, 1, 0, 0, 0, 1, 0],  # x10
        [1, 1, 0, 1, 0, 0, -1]  # x11
    ]
    v2_rows = [
        [1, 1, 0, -1, 0, 1, 0], # y00
        [0, 0, 1, 0, 0, 1, 0],  # y01
        [0, 0, 0, 1, 0, 0, 1],  # y10
        [1, 0, -1, 0, 1, 0, 1]  # y11
    ]
    w2_rows = [
        [1, 0, 0, 1, -1, 0, 1], # z00
        [0, 0, 1, 0, 1, 0, 0],  # z01
        [0, 1, 0, 1, 0, 0, 0],  # z10
        [1, -1, 1, 0, 0, 1, 0]  # z11
    ]
    
    U2 = np.array(u2_rows)
    V2 = np.array(v2_rows)
    W2 = np.array(w2_rows)
    
    # Block Strassen Mapping for 4x4x4
    # A, B, C are 4x4. Split into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B
    
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)B00
    # P2 = A00(B01-B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P5 = (A10-A00)(B00+B01)
    # P6 = (A01-A11)(B10+B11)
    
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    # We need to map the 7 Strassen products P_k (each rank 7 from 2x2x2)
    # to the 4x4 blocks.
    
    # For each of the 2 halves of B (cols 0-3, 4-7), we apply the 4x4x4 Strassen.
    # Half 0: B[:, 0:4]
    # Half 1: B[:, 4:8]
    
    # Pre-calculate indices for U, V, W for one 4x4x4 block
    # U_block (16x49), V_block (32x49), W_block (16x49) -- Note W is output C (4x4) so 16 entries
    
    U_block = np.zeros((16, 49), dtype=int)
    V_block = np.zeros((32, 49), dtype=int) # V is m*p = 4*8 = 32? No, for one half B is 4x4, so V_half is 16x49.
    # Wait, V shape is (m*p, rank) = (32, 98).
    # For one half, V_half is (16, 49).
    V_half_template = np.zeros((16, 49), dtype=int)
    W_half_template = np.zeros((16, 49), dtype=int) # C is 4x4, so 16 entries.
    
    # Mapping for 4x4 Strassen
    # A_blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Each block is 2x2.
    
    # Specs for A_eff, B_eff, C_eff
    # (coeff, block_index)
    A_specs = [
        [(1, 0), (1, 3)], # P0
        [(1, 2), (1, 3)], # P1
        [(1, 0)],         # P2
        [(1, 3)],         # P3
        [(1, 0), (1, 1)], # P4
        [(1, 2), (-1, 0)],# P5
        [(1, 1), (-1, 3)] # P6
    ]
    B_specs = [
        [(1, 0), (1, 3)], # P0
        [(1, 0)],         # P1
        [(1, 1), (-1, 3)],# P2
        [(1, 2), (-1, 0)],# P3
        [(1, 3)],         # P4
        [(1, 0), (1, 1)], # P5
        [(1, 2), (1, 3)]  # P6
    ]
    C_specs = [
        [(1, 0), (1, 3)], # P0 -> C00, C11
        [(1, 2), (-1, 3)],# P1 -> C10, -C11
        [(1, 1), (1, 3)], # P2 -> C01, C11
        [(1, 0), (1, 2)], # P3 -> C00, C10
        [(-1, 0), (1, 1)],# P4 -> -C00, C01
        [(1, 3)],         # P5 -> C11
        [(1, 0)]          # P6 -> C00
    ]
    
    for k in range(7):
        for ms in range(7):
            col_idx = k * 7 + ms
            
            # U
            for coeff, blk in A_specs[k]:
                bi, bj = divmod(blk, 2)
                # Block (bi, bj) of A (4x4) corresponds to rows bi*2..bi*2+1, cols bj*2..bj*2+1
                # U2 maps 2x2 A_eff to 7 scalars.
                # A_eff index r,c maps to global A index (bi*2+r, bj*2+c)
                for r in range(2):
                    for c in range(2):
                        u_idx = r * 2 + c
                        val = U2[u_idx, ms]
                        if val != 0:
                            a_row = bi * 2 + r
                            a_col = bj * 2 + c
                            global_a_idx = a_row * 4 + a_col
                            U_block[global_a_idx, col_idx] += coeff * val
            
            # V (for one half of B, 4x4)
            for coeff, blk in B_specs[k]:
                bi, bj = divmod(blk, 2)
                for r in range(2):
                    for c in range(2):
                        v_idx = r * 2 + c
                        val = V2[v_idx, ms]
                        if val != 0:
                            b_row = bi * 2 + r
                            b_col = bj * 2 + c
                            global_b_idx = b_row * 4 + b_col # Local to 4x4 block
                            V_half_template[global_b_idx, col_idx] += coeff * val
            
            # W (Output C, 4x4)
            for coeff, blk in C_specs[k]:
                bi, bj = divmod(blk, 2)
                for r in range(2):
                    for c in range(2):
                        w_idx = r * 2 + c
                        val = W2[w_idx, ms]
                        if val != 0:
                            c_row = bi * 2 + r
                            c_col = bj * 2 + c
                            global_c_idx = c_row * 4 + c_col
                            W_half_template[global_c_idx, col_idx] += coeff * val
            
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fill U, V, W for both halves
    # U is same for both halves (A is 4x4, doesn't change)
    # U (16x98)
    U[:, 0:49] = U_block
    U[:, 49:98] = U_block
    
    # V (32x98)
    # V rows 0-15 correspond to B[:, 0:4]
    # V rows 16-31 correspond to B[:, 4:8]
    V[0:16, 0:49] = V_half_template
    V[16:32, 0:49] = 0 # Wait, V_half_template is for B_left (cols 0-3).
    # For the first half (cols 0-3), V uses rows 0-15.
    # For the second half (cols 4-7), V uses rows 16-31.
    
    # Correction:
    # V[0:16, 0:49] should be V_half_template (for B_left)
    # V[16:32, 49:98] should be V_half_template (for B_right)
    # The off-diagonals are 0.
    
    V[0:16, 0:49] = V_half_template
    V[16:32, 49:98] = V_half_template
    
    # W (32x98)
    # W index: k*n + i. k is col of C (0-7), i is row of C (0-3).
    # C is 4x8.
    # C[:, 0:4] comes from P_0..P_6 of half 0.
    # C[:, 4:8] comes from P_0..P_6 of half 1.
    
    # W_half_template is 16x49. Indices are (c_row, c_col) in 4x4.
    # For Half 0 (cols 0-3):
    # W rows correspond to k=0..3.
    # For each k, i=0..3.
    # global_w_idx = k * 4 + i.
    # This matches the index used in W_half_template (c_row * 4 + c_col).
    # So W[0:16, 0:49] = W_half_template.
    
    # For Half 1 (cols 4-7):
    # k=4..7.
    # global_w_idx = k * 4 + i = (4 + c_col) * 4 + c_row = 16 + c_col * 4 + c_row.
    # So W[16:32, 49:98] = W_half_template.
    
    W[0:16, 0:49] = W_half_template
    W[16:32, 49:98] = W_half_template
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
