import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strategy: Strassen 4x4x4 (rank 49) for left 4 cols of B
    #           + Schoolbook 4x4x3 (rank 48) for right 3 cols of B
    # Total rank = 97
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97
    
    # Strassen 2x2x2 base coefficients
    u_in = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_in = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_out = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    col = 0
    for k_outer in range(7):
        for l_inner in range(7):
            if time.time() > deadline:
                break
            
            u_vec = [0] * 16
            for i_in in range(2):
                for j_in in range(2):
                    coeff_in = u_in[l_inner][i_in * 2 + j_in]
                    if coeff_in == 0:
                        continue
                    for b_out in range(4):
                        coeff_out = u_in[k_outer][b_out]
                        if coeff_out == 0:
                            continue
                        row_block = b_out // 2
                        col_block = b_out % 2
                        i_global = 2 * row_block + i_in
                        j_global = 2 * col_block + j_in
                        u_vec[i_global * 4 + j_global] += coeff_in * coeff_out
            
            v_vec = [0] * 28
            for i_in in range(2):
                for j_in in range(2):
                    coeff_in = v_in[l_inner][i_in * 2 + j_in]
                    if coeff_in == 0:
                        continue
                    for b_out in range(4):
                        coeff_out = v_in[k_outer][b_out]
                        if coeff_out == 0:
                            continue
                        row_block = b_out // 2
                        col_block = b_out % 2
                        j_global = 2 * row_block + i_in
                        k_global = 2 * col_block + j_in
                        v_vec[j_global * 7 + k_global] += coeff_in * coeff_out
            
            w_vec = [0] * 28
            for c_block_out in range(4):
                coeff_w_out = w_out[c_block_out][k_outer]
                if coeff_w_out == 0:
                    continue
                for c_block_in in range(4):
                    coeff_w_in = w_out[c_block_in][l_inner]
                    if coeff_w_in == 0:
                        continue
                    total_coeff = coeff_w_out * coeff_w_in
                    if total_coeff == 0:
                        continue
                    
                    c_row_block = c_block_out // 2
                    c_col_block = c_block_out % 2
                    c_inner_row = c_block_in // 2
                    c_inner_col = c_block_in % 2
                    
                    i_global = 2 * c_row_block + c_inner_row
                    k_global = 2 * c_col_block + c_inner_col
                    
                    w_vec[k_global * 4 + i_global] += total_coeff
            
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]
            
            col += 1
        
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                if time.time() > deadline:
                    break
                k_global = 4 + k_prime
                U[i * 4 + j][col] = 1
                V[j * 7 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
