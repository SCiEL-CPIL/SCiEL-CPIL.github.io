import numpy as np

def _strassen_2x2x2():
    """
    Returns U, V, W matrices for 2x2x2 Strassen decomposition.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Indices for 2x2 matrix flattened row-major: 00, 01, 10, 11
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # P1 = (A00 + A11)(B00 + B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # P2 = (A10 + A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] -= 1
    
    # P3 = A00(B01 - B11)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1
    
    # P4 = A11(B10 - B00)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1
    
    # P5 = (A00 + A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] += 1
    
    # P6 = (A10 - A00)(B00 + B01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # P7 = (A01 - A11)(B10 + B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W

def _strassen_4x4x4():
    """
    Returns U, V, W for 4x4x4 Strassen decomposition.
    Rank 49.
    """
    U_2, V_2, W_2 = _strassen_2x2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # Block definitions for 4x4
    # A blocks: 00 (0-1, 0-1), 01 (0-1, 2-3), 10 (2-3, 0-1), 11 (2-3, 2-3)
    # B blocks: same
    # C blocks: same
    
    # Strassen terms for 4x4
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Each P is a 2x2 matrix product, decomposed into 7 rank-1 terms.
    # Total 49 terms.
    
    # Helper to add block contributions
    def add_block(matrix, block_idx, r_offset, c_offset, coeff, base_mat):
        # block_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        bi, bj = divmod(block_idx, 2)
        # base_mat is 4x7 (for U, V) or 4x7 (for W)
        # We need to map the 2x2 local indices to 4x4 global indices
        for lr in range(2):
            for lc in range(2):
                global_r = bi * 2 + lr
                global_c = bj * 2 + lc
                global_idx = global_r * 4 + global_c
                local_idx = lr * 2 + lc
                matrix[global_idx, r_offset:c_offset] += coeff * base_mat[local_idx, :]
                
    # This helper is a bit slow for large matrices, but 4x4 is small.
    # Better to precompute the 4x4 base matrices for each Strassen term.
    
    # Let's generate the 4x4 U, V, W directly by iterating the 7 top-level terms.
    
    # Top-level term definitions: (coeff, a_block, b_block, c_block)
    # Actually, we just need to apply the 2x2 decomposition to the effective blocks.
    
    # Effective A and B for each of the 7 top-level P_k
    A_eff_specs = [
        [(1, 0), (1, 3)], # P1: A00+A11
        [(1, 2), (1, 3)], # P2: A10+A11
        [(1, 0)],         # P3: A00
        [(1, 3)],         # P4: A11
        [(1, 0), (1, 1)], # P5: A00+A01
        [(1, 2), (-1, 0)],# P6: A10-A00
        [(1, 1), (-1, 3)] # P7: A01-A11
    ]
    
    B_eff_specs = [
        [(1, 0), (1, 3)], # P1: B00+B11
        [(1, 0)],         # P2: B00
        [(1, 1), (-1, 3)],# P3: B01-B11
        [(1, 2), (-1, 0)],# P4: B10-B00
        [(1, 3)],         # P5: B11
        [(1, 0), (1, 1)], # P6: B00+B01
        [(1, 2), (1, 3)]  # P7: B10+B11
    ]
    
    C_eff_specs = [
        [(1, 0), (1, 3)], # C00+C11
        [(1, 2), (-1, 3)],# C10-C11
        [(1, 1), (1, 3)], # C01+C11
        [(1, 0), (1, 2)], # C00+C10
        [(-1, 0), (1, 1)],# -C00+C01
        [(1, 3)],         # C11
        [(1, 0)]          # C00
    ]
    
    for k in range(7):
        for m in range(7):
            col_idx = k * 7 + m
            
            # U contribution
            for coeff_a, a_blk in A_eff_specs[k]:
                bi, bj = divmod(a_blk, 2)
                for lr in range(2):
                    for lc in range(2):
                        gr = bi * 2 + lr
                        gc = bj * 2 + lc
                        g_idx = gr * 4 + gc
                        l_idx = lr * 2 + lc
                        U[g_idx, col_idx] += coeff_a * U_2[l_idx, m]
                        
            # V contribution
            for coeff_b, b_blk in B_eff_specs[k]:
                bi, bj = divmod(b_blk, 2)
                for lr in range(2):
                    for lc in range(2):
                        gr = bi * 2 + lr
                        gc = bj * 2 + lc
                        g_idx = gr * 4 + gc
                        l_idx = lr * 2 + lc
                        V[g_idx, col_idx] += coeff_b * V_2[l_idx, m]
                        
            # W contribution
            for coeff_c, c_blk in C_eff_specs[k]:
                bi, bj = divmod(c_blk, 2)
                for lr in range(2):
                    for lc in range(2):
                        gr = bi * 2 + lr
                        gc = bj * 2 + lc
                        # W index for 4x4 is col*4 + row
                        g_idx = gc * 4 + gr
                        l_idx = lr * 2 + lc
                        W[g_idx, col_idx] += coeff_c * W_2[l_idx, m]
                        
    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Get 4x4x4 decomposition
    U_4, V_4, W_4 = _strassen_4x4x4()
    
    # Apply to Left half (B cols 0-3)
    for i in range(16):
        U[i, 0:49] = U_4[i, :]
        
    for i in range(16):
        # V index for B: row * 8 + col
        # Left half: cols 0-3
        # B_4 index in U_4/V_4 is row * 4 + col
        # We need to map B_4 (4x4) to B (4x8)
        # V_4 has indices for 4x4 B.
        # Global V index = r * 8 + c
        # V_4 index = r * 4 + c
        r, c = divmod(i, 4)
        v_idx_left = r * 8 + c
        V[v_idx_left, 0:49] = V_4[i, :]
        
    for i in range(16):
        # W index for C: col * 4 + row
        # W_4 index = col * 4 + row
        # Left half C: cols 0-3
        # Global W index = k * 4 + i
        # k is 0-3 in left half.
        # W_4 index k' is 0-3.
        # Global k = k'
        W[i, 0:49] = W_4[i, :]
        
    # Apply to Right half (B cols 4-7)
    for i in range(16):
        U[i, 49:98] = U_4[i, :]
        
    for i in range(16):
        # V index for B: row * 8 + col
        # Right half: cols 4-7
        # V_4 index = r * 4 + c
        r, c = divmod(i, 4)
        v_idx_right = r * 8 + (c + 4)
        V[v_idx_right, 49:98] = V_4[i, :]
        
    for i in range(16):
        # W index for C: col * 4 + row
        # Right half C: cols 4-7
        # W_4 index k' is 0-3.
        # Global k = k' + 4
        # Global W index = (k' + 4) * 4 + i_row
        # W_4 index = k' * 4 + i_row
        # So Global W index = W_4 index + 16
        w_idx_global = i + 16
        W[w_idx_global, 49:98] = W_4[i, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
