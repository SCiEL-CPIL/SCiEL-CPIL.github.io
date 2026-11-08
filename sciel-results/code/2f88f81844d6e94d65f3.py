import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Hybrid: Strassen 4x4x4 (rank 49) + Schoolbook 4x4x3 (rank 48) = 97
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 coefficients
    u_in = [
        [1, 0, 0, 1],   # M1: (A00+A11)
        [0, 0, 1, 1],   # M2: (A10+A11)
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: (A00+A01)
        [-1, 0, 1, 0],  # M6: (A10-A00)
        [0, 1, 0, -1]   # M7: (A11-A01)
    ]
    v_in = [
        [1, 0, 0, 1],   # M1: (B00+B11)
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: (B10-B00)
        [-1, 0, 1, 0],  # M4: (B01-B11)
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: (B00+B01)
        [0, 0, 1, 1]    # M7: (B10+B11)
    ]
    w_out = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 = M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],   # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C10 = M2 + M4
        [1, -1, 1, 0, 0, 1, 0]   # C11 = M1 - M2 + M3 + M6
    ]
    
    # Outer Strassen 4x4x4 uses same structure
    u_out = u_in
    v_out = v_in
    w_outer = w_out
    
    col = 0
    for k in range(7):
        for l in range(7):
            if time.time() > deadline:
                # Return what we have; the construction is deterministic and fast
                # so this should never trigger for the 49 iterations
                pass
            
            # U vector: 16 elements, index = i*4 + j
            u_vec = [0] * 16
            for ii in range(2):
                for jj in range(2):
                    ci = u_in[l][ii * 2 + jj]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = u_out[k][b]
                        if co == 0:
                            continue
                        b_row = b // 2
                        b_col = b % 2
                        i_g = 2 * b_row + ii
                        j_g = 2 * b_col + jj
                        u_vec[i_g * 4 + j_g] += ci * co
            
            # V vector: 28 elements, index = j*7 + k
            v_vec = [0] * 28
            for ii in range(2):
                for jj in range(2):
                    ci = v_in[l][ii * 2 + jj]
                    if ci == 0:
                        continue
                    for b in range(4):
                        co = v_out[k][b]
                        if co == 0:
                            continue
                        b_row = b // 2
                        b_col = b % 2
                        j_g = 2 * b_row + ii
                        k_g = 2 * b_col + jj
                        # Only first 4 columns of B (indices 0-3)
                        v_vec[j_g * 7 + k_g] += ci * co
            
            # W vector: 28 elements, index = k*4 + i
            w_vec = [0] * 28
            for cb in range(4):
                co_w = w_outer[cb][k]
                if co_w == 0:
                    continue
                for ci_idx in range(4):
                    ci_w = w_out[ci_idx][l]
                    if ci_w == 0:
                        continue
                    total = co_w * ci_w
                    if total == 0:
                        continue
                    cb_row = cb // 2
                    cb_col = cb % 2
                    ci_row = ci_idx // 2
                    ci_col = ci_idx % 2
                    i_g = 2 * cb_row + ci_row
                    k_g = 2 * cb_col + ci_col
                    w_vec[k_g * 4 + i_g] += total
            
            for r in range(16):
                U[r][col] = u_vec[r]
            for r in range(28):
                V[r][col] = v_vec[r]
            for r in range(28):
                W[r][col] = w_vec[r]
            
            col += 1
    
    # Part 2: 4x4 * 4x3 schoolbook, rank 48
    for i in range(4):
        for j in range(4):
            for kp in range(3):
                if time.time() > deadline:
                    break
                kg = 4 + kp
                U[i * 4 + j][col] = 1
                V[j * 7 + kg][col] = 1
                W[kg * 4 + i][col] = 1
                col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
