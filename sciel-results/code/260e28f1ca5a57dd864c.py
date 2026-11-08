import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x7 matrix.
    
    Interface: construct() -> {'U', 'V', 'W'}
    Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    Convention: W[k*n+i, r]
    
    Strategy:
    Split the multiplication into two independent parts:
    1. Left part: A (4x4) * B_left (4x4) -> C_left (4x4)
       Use Strassen's algorithm for 4x4x4 with rank 49.
       
    2. Right part: A (4x4) * B_right (4x3) -> C_right (4x3)
       Use schoolbook multiplication with rank 4*4*3 = 48.
       
    Total rank = 49 + 48 = 97.
    """
    
    start_time = time.time()
    deadline = start_time + 110.0 # 120s limit, 10s safety margin

    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
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
    w_outer = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    rank_strassen_4x4 = 49
    rank_schoolbook_4x4x3 = 4 * 4 * 3  # 48
    rank = rank_strassen_4x4 + rank_schoolbook_4x4x3  # 97
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97
    
    # --- Part 1: 4x4 * 4x4 (Left part of B, cols 0-3) ---
    col = 0
    for k_outer in range(7):
        for l_inner in range(7):
            # Calculate U vector (16 elements)
            # U_idx = i * m + j = i * 4 + j
            u_vec = [0] * 16
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = u_in[l_inner][i_inner * 2 + j_inner]
                    if coeff_inner == 0:
                        continue
                    for b in range(4):
                        coeff_outer = u_out[k_outer][b]
                        if coeff_outer == 0:
                            continue
                        # Block b maps to:
                        # b=0: rows 0-1, cols 0-1
                        # b=1: rows 0-1, cols 2-3
                        # b=2: rows 2-3, cols 0-1
                        # b=3: rows 2-3, cols 2-3
                        b_row = b // 2
                        b_col = b % 2
                        i_global = 2 * b_row + i_inner
                        j_global = 2 * b_col + j_inner
                        u_vec[i_global * 4 + j_global] += coeff_inner * coeff_outer
            
            # Calculate V vector (28 elements, but only first 16 are non-zero for this part)
            # V_idx = j * p + k = j * 7 + k
            v_vec = [0] * 28
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = v_in[l_inner][i_inner * 2 + j_inner]
                    if coeff_inner == 0:
                        continue
                    for b in range(4):
                        coeff_outer = v_out[k_outer][b]
                        if coeff_outer == 0:
                            continue
                        b_row = b // 2
                        b_col = b % 2
                        j_global = 2 * b_row + i_inner
                        k_global = 2 * b_col + j_inner
                        # B is 4x7. Left part is cols 0-3.
                        # V index is j_global * 7 + k_global
                        v_vec[j_global * 7 + k_global] += coeff_inner * coeff_outer
            
            # Calculate W vector (28 elements, but only first 16 are non-zero for this part)
            # W_idx = k * n + i = k * 4 + i
            w_vec = [0] * 28
            for c_block in range(4):
                coeff_outer_w = w_outer[c_block][k_outer]
                if coeff_outer_w == 0:
                    continue
                for c_inner in range(4):
                    coeff_inner_w = w_out[c_inner][l_inner]
                    if coeff_inner_w == 0:
                        continue
                    total_coeff = coeff_outer_w * coeff_inner_w
                    if total_coeff == 0:
                        continue
                    
                    c_block_row = c_block // 2
                    c_block_col = c_block % 2
                    c_inner_row = c_inner // 2
                    c_inner_col = c_inner % 2
                    
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
            
            if time.time() > deadline:
                # Return partial result if deadline exceeded (should not happen)
                return {"U": U, "V": V, "W": W}
            
    # --- Part 2: 4x4 * 4x3 (Right part of B, cols 4-6) ---
    # Schoolbook Rank 48
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                k_global = 4 + k_prime
                U[i * 4 + j][col] = 1
                V[j * 7 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1
                
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
