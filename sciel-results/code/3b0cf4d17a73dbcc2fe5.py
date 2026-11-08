import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 135s of 150s budget, leaving 15s safety margin

    # We use a hybrid decomposition:
    # Part 1: Strassen 4x4x4 (rank 49) for A(4x4) * B_left(4x4)
    # Part 2: Schoolbook 4x4x3 (rank 48) for A(4x4) * B_right(4x3)
    # Total rank = 49 + 48 = 97
    
    rank = 97
    U = np.zeros((n * m, rank))  # (16, 97)
    V = np.zeros((m * p, rank))  # (28, 97)
    W = np.zeros((p * n, rank))  # (28, 97)
    
    # Strassen 2x2x2 base coefficients
    # For 2x2 * 2x2 -> 2x2
    # u_in[l][i,j], v_in[l][i,j], w_out[c_block][l]
    # Blocks indexed as 0=00, 1=01, 2=10, 3=11
    u_in = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    v_in = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    w_out = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    # Outer Strassen for 4x4x4: same structure applied to 2x2 blocks
    u_out = u_in  # Same coefficients
    v_out = v_in  # Same coefficients
    w_outer = w_out  # Same coefficients
    
    col = 0
    for k_outer in range(7):  # outer index for the 4x4 block structure
        for l_inner in range(7):  # inner index for the 2x2 block structure
            if time.time() > deadline:
                break
            
            # Build U vector (16 elements: i*4+j)
            u_vec = np.zeros(16)
            for i_inner in range(2):
                for j_inner in range(2):
                    coeff_inner = u_in[l_inner][i_inner * 2 + j_inner]
                    if coeff_inner == 0:
                        continue
                    for b in range(4):
                        coeff_outer = u_out[k_outer][b]
                        if coeff_outer == 0:
                            continue
                        b_row = b // 2
                        b_col = b % 2
                        i_global = 2 * b_row + i_inner
                        j_global = 2 * b_col + j_inner
                        u_vec[i_global * 4 + j_global] += coeff_inner * coeff_outer
            
            # Build V vector (28 elements: j*7+k)
            # Only the left 4 columns of B (indices 0-3) are used in this part
            v_vec = np.zeros(28)
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
                        # k_global is 0-3, which corresponds to the left part of B
                        v_vec[j_global * 7 + k_global] += coeff_inner * coeff_outer
            
            # Build W vector (28 elements: k*4+i)
            w_vec = np.zeros(28)
            for c_block_outer in range(4):
                coeff_outer_w = w_outer[c_block_outer][k_outer]
                if coeff_outer_w == 0:
                    continue
                for c_inner in range(4):
                    coeff_inner_w = w_out[c_inner][l_inner]
                    if coeff_inner_w == 0:
                        continue
                    total_coeff = coeff_outer_w * coeff_inner_w
                    if total_coeff == 0:
                        continue
                    
                    c_block_row = c_block_outer // 2
                    c_block_col = c_block_outer % 2
                    c_inner_row = c_inner // 2
                    c_inner_col = c_inner % 2
                    
                    i_global = 2 * c_block_row + c_inner_row
                    k_global = 2 * c_block_col + c_inner_col
                    
                    # W index: k_global * n + i_global = k_global * 4 + i_global
                    w_vec[k_global * 4 + i_global] += total_coeff
            
            U[0:16, col] = u_vec
            V[0:28, col] = v_vec
            W[0:28, col] = w_vec
            col += 1
        
        if time.time() > deadline:
            break
    
    # Part 2: Schoolbook 4x4 * 4x3 (right part of B, columns 4-6)
    for i in range(4):
        for j in range(4):
            for k_prime in range(3):
                k_global = 4 + k_prime
                U[i * 4 + j, col] = 1.0
                V[j * 7 + k_global, col] = 1.0
                W[k_global * 4 + i, col] = 1.0
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Convert to list format as required
    U_list = [[U[r, c] for c in range(rank)] for r in range(n * m)]
    V_list = [[V[r, c] for c in range(rank)] for r in range(m * p)]
    W_list = [[W[r, c] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
