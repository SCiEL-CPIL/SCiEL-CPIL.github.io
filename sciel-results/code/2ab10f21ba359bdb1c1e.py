import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Target Rank: 49 (Strassen 4x4x4) + 48 (Schoolbook 4x4x3) = 97
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4 * 4x4 (Left part of B) ---
    # Strassen 4x4x4 (Rank 49)
    
    # Inner Strassen 2x2x2 coefficients
    # u_in[l] = coeffs for A blocks (0:00, 1:01, 2:10, 3:11)
    # v_in[l] = coeffs for B blocks
    u_in = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    v_in = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    # w_out[c_block][m_idx]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    w_out = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Outer Strassen 4x4x4 coefficients
    # u_out[k] = coeffs for A blocks (0:00, 1:01, 2:10, 3:11)
    # v_out[k] = coeffs for B blocks
    u_out = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    v_out = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    # w_outer[c_block][k_idx]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    w_outer = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    col = 0
    # Iterate over the 49 rank-1 terms for the 4x4x4 part
    # k indexes the outer Strassen step (0-6)
    # l indexes the inner Strassen step (0-6)
    for k in range(7):
        for l in range(7):
            # Calculate U vector (16 elements)
            # U_idx = i * m + j = i * 4 + j
            u_vec = [0] * 16
            # u_out[k] gives coefficients for the 4 blocks of A in the outer step
            # u_in[l] gives coefficients for the 4 blocks of A in the inner step
            # The blocks of A in the outer step are 2x2 matrices.
            # Let A be partitioned into 4 blocks: A00, A01, A10, A11.
            # u_out[k] = [c00, c01, c10, c11] such that A_outer = c00*A00 + c01*A01 + c10*A10 + c11*A11
            # Then for each block Abc, we apply u_in[l] to its 2x2 sub-blocks.
            
            for b in range(4):
                coeff_outer = u_out[k][b]
                if coeff_outer == 0: continue
                
                # Block b corresponds to:
                # b=0: rows 0-1, cols 0-1 (A00)
                # b=1: rows 0-1, cols 2-3 (A01)
                # b=2: rows 2-3, cols 0-1 (A10)
                # b=3: rows 2-3, cols 2-3 (A11)
                
                b_row = b // 2
                b_col = b % 2
                
                for i_inner in range(2):
                    for j_inner in range(2):
                        coeff_inner = u_in[l][i_inner * 2 + j_inner]
                        if coeff_inner == 0: continue
                        
                        # Global indices
                        i_global = 2 * b_row + i_inner
                        j_global = 2 * b_col + j_inner
                        
                        u_vec[i_global * 4 + j_global] += coeff_outer * coeff_inner
            
            # Calculate V vector (28 elements)
            # V_idx = j * p + k = j * 7 + k
            # Only the first 4 columns of B (indices 0-3) are involved in this part.
            v_vec = [0] * 28
            for b in range(4):
                coeff_outer = v_out[k][b]
                if coeff_outer == 0: continue
                
                b_row = b // 2
                b_col = b % 2
                
                for i_inner in range(2):
                    for j_inner in range(2):
                        # v_in[l] corresponds to B blocks.
                        # In Strassen, V coefficients match U coefficients structure for the second matrix.
                        # v_in[l] = [c00, c01, c10, c11] for B blocks.
                        coeff_inner = v_in[l][i_inner * 2 + j_inner]
                        if coeff_inner == 0: continue
                        
                        # Global indices for B
                        j_global = 2 * b_row + i_inner
                        k_global = 2 * b_col + j_inner
                        
                        # B is 4x7. Left part is cols 0-3.
                        # V index is j_global * 7 + k_global
                        v_vec[j_global * 7 + k_global] += coeff_outer * coeff_inner
            
            # Calculate W vector (28 elements)
            # W_idx = k * n + i = k * 4 + i
            # Only the first 4 rows of C (indices 0-3) are involved in this part.
            w_vec = [0] * 28
            
            # w_outer[c_block][k] gives coeff for outer result block
            # w_out[c_inner][l] gives coeff for inner result block
            
            for c_block in range(4):
                coeff_outer_w = w_outer[c_block][k]
                if coeff_outer_w == 0: continue
                
                c_block_row = c_block // 2
                c_block_col = c_block % 2
                
                for c_inner in range(4):
                    coeff_inner_w = w_out[c_inner][l]
                    if coeff_inner_w == 0: continue
                    
                    total_coeff = coeff_outer_w * coeff_inner_w
                    if total_coeff == 0: continue
                    
                    c_inner_row = c_inner // 2
                    c_inner_col = c_inner % 2
                    
                    # Global indices for C
                    i_global = 2 * c_block_row + c_inner_row
                    k_global = 2 * c_block_col + c_inner_col
                    
                    # W index is k_global * n + i_global = k_global * 4 + i_global
                    w_vec[k_global * 4 + i_global] += total_coeff
            
            # Add to U, V, W
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]
            
            col += 1
            
            # Check time
            if time.time() > deadline:
                # Return best effort so far
                return {"U": U, "V": V, "W": W}
    
    # --- Part 2: 4x4 * 4x3 (Right part of B) ---
    # Schoolbook Rank 48
    # A is 4x4, B_right is 4x3 (columns 4,5,6 of the full 4x7 B)
    # C_right is 4x3 (columns 4,5,6 of the full 4x7 C)
    
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                k_global = 4 + k_prime
                U[i * 4 + j][col] = 1
                V[j * 7 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
