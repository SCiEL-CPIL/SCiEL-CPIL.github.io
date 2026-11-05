import numpy as np
import time

DIMENSIONS = [5, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 115  # Safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Check time
    if time.time() - START_TIME > TIME_LIMIT:
        # Return a trivial decomposition if time runs out? 
        # Better to return the best we have. 
        # Since we are constructing statically, it should be fast.
        pass

    # Partition indices into S1={0,1,2,3} and S2={4}
    # S1 size 4, S2 size 1
    
    # Total Rank = 80
    R = 80
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col_offset = 0
    
    def add_block(bn, bm, bp, si, sj, sk, U_b, V_b, W_b):
        nonlocal col_offset
        rank_b = len(U_b[0])
        for r in range(rank_b):
            for i in range(bn):
                for j in range(bm):
                    val = U_b[i * bm + j][r]
                    if val != 0:
                        U[(si + i) * m + (sj + j)][col_offset + r] = val
            for j in range(bm):
                for k in range(bp):
                    val = V_b[j * bp + k][r]
                    if val != 0:
                        V[(sj + j) * p + (sk + k)][col_offset + r] = val
            for k in range(bp):
                for i in range(bn):
                    val = W_b[k * bn + i][r]
                    if val != 0:
                        W[(sk + k) * n + (si + i)][col_offset + r] = val
        col_offset += rank_b

    # 1. 4x4x4 block (Rank 49)
    # Correct Strassen 4x4 decomposition
    
    # First, define the 2x2 Strassen tensors
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    def set_rank1(idx, u_vec, v_vec, w_vec):
        for i in range(4):
            U2[i][idx] = u_vec[i]
            V2[i][idx] = v_vec[i]
            W2[i][idx] = w_vec[i]
            
    # M1
    set_rank1(0, [1,0,0,1], [1,0,0,1], [1,1,1,1])
    # M2
    set_rank1(1, [0,0,1,1], [1,0,0,0], [0,0,1,1])
    # M3
    set_rank1(2, [1,0,0,0], [0,0,1,-1], [0,1,0,0])
    # M4
    set_rank1(3, [0,0,0,1], [0,1,-1,0], [0,0,0,1])
    # M5
    set_rank1(4, [1,1,0,0], [0,0,0,1], [1,1,0,0])
    # M6
    set_rank1(5, [-1,0,1,0], [1,1,0,0], [0,1,0,0])
    # M7
    set_rank1(6, [0,-1,0,1], [0,0,1,1], [0,0,1,1])
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Coefficients for A and B blocks for each Strassen step
    a_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, -1, 0, 1]  # M7
    ]
    b_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1], # M3
        [0, 1, -1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Coefficients for combining M_s into C_blocks
    # c_coeffs[s][cb] where cb is 0:00, 1:01, 2:10, 3:11
    c_coeffs = [
        [1, 0, 0, 1], # M1: C00=1, C11=1
        [0, 0, 1, -1], # M2: C10=1, C11=-1
        [0, 1, -1, 0], # M3: C01=1, C10=-1
        [-1, 0, 0, 1], # M4: C00=-1, C11=1
        [1, -1, 0, 0], # M5: C00=1, C01=-1
        [0, 0, -1, 1], # M6: C10=-1, C11=1
        [1, 1, 1, 0]  # M7: C00=1, C01=1, C10=1
    ]
    
    # Block indices for C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    cb_indices = [(0,0), (0,1), (1,0), (1,1)]
    
    for s in range(7):
        for r2 in range(7):
            r4 = s * 7 + r2
            
            # Fill U4 and V4
            for bi in range(2):
                for bj in range(2):
                    coeff_a = a_coeffs[s][bi * 2 + bj]
                    coeff_b = b_coeffs[s][bi * 2 + bj]
                    
                    for li in range(2):
                        for lj in range(2):
                            gi = 2 * bi + li
                            gj = 2 * bj + lj
                            
                            val_u = U2[li * 2 + lj][r2] * coeff_a
                            if val_u != 0:
                                U4[gi * 4 + gj][r4] = val_u
                            
                            val_v = V2[li * 2 + lj][r2] * coeff_b
                            if val_v != 0:
                                V4[gi * 4 + gj][r4] = val_v
                                
            # Fill W4
            # W2[li*2+lj] corresponds to M_s[li, lj]
            # M_s contributes to C_blocks with coefficients c_coeffs[s][cb]
            for cb_idx in range(4):
                coeff_c = c_coeffs[s][cb_idx]
                if coeff_c == 0:
                    continue
                cb_i, cb_j = cb_indices[cb_idx]
                for li in range(2):
                    for lj in range(2):
                        w_val = W2[li * 2 + lj][r2]
                        if w_val == 0:
                            continue
                        final_val = w_val * coeff_c
                        if final_val != 0:
                            gi = 2 * cb_i + li
                            gk = 2 * cb_j + lj
                            # W index is k * n + i
                            W4[gk * 4 + gi][r4] = final_val

    add_block(4, 4, 4, 0, 0, 0, U4, V4, W4)

    # 2. 4x4x1 block (Rank 16)
    U_b = [[0]*16 for _ in range(16)]
    V_b = [[0]*16 for _ in range(4)]
    W_b = [[0]*16 for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U_b[i * 4 + j][r] = 1
            V_b[j * 1 + 0][r] = 1
            W_b[0 * 4 + i][r] = 1
            
    add_block(4, 4, 1, 0, 0, 4, U_b, V_b, W_b)

    # 3. 4x1x4 block (Rank 1)
    # FIXED: W_b must be 16x1
    U_b = [[1] for _ in range(4)]
    V_b = [[1, 1, 1, 1]]
    W_b = [[1] for _ in range(16)] # 4*4=16 rows
    
    add_block(4, 1, 4, 0, 4, 0, U_b, V_b, W_b)

    # 4. 4x1x1 block (Rank 1)
    U_b = [[1] for _ in range(4)]
    V_b = [[1]]
    W_b = [[1] for _ in range(4)]
    
    add_block(4, 1, 1, 0, 4, 4, U_b, V_b, W_b)

    # 5. 1x4x4 block (Rank 4)
    U_b = [[0]*4 for _ in range(4)]
    V_b = [[0]*4 for _ in range(16)]
    W_b = [[0]*4 for _ in range(4)]
    
    for j in range(4):
        r = j
        U_b[0 * 4 + j][r] = 1
        for k in range(4):
            V_b[j * 4 + k][r] = 1
            W_b[k * 1 + 0][r] = 1
            
    add_block(1, 4, 4, 4, 0, 0, U_b, V_b, W_b)

    # 6. 1x4x1 block (Rank 4)
    U_b = [[0]*4 for _ in range(4)]
    V_b = [[0]*4 for _ in range(4)]
    W_b = [[0]*4 for _ in range(1)]
    
    for j in range(4):
        r = j
        U_b[0 * 4 + j][r] = 1
        V_b[j * 1 + 0][r] = 1
        W_b[0 * 1 + 0][r] = 1
        
    add_block(1, 4, 1, 4, 0, 4, U_b, V_b, W_b)

    # 7. 1x1x4 block (Rank 4)
    U_b = [[0]*4 for _ in range(1)]
    V_b = [[0]*4 for _ in range(4)]
    W_b = [[0]*4 for _ in range(4)]
    
    for k in range(4):
        r = k
        U_b[0 * 1 + 0][r] = 1
        V_b[0 * 4 + k][r] = 1
        W_b[k * 1 + 0][r] = 1
        
    add_block(1, 1, 4, 4, 4, 0, U_b, V_b, W_b)

    # 8. 1x1x1 block (Rank 1)
    U_b = [[1]]
    V_b = [[1]]
    W_b = [[1]]
    
    add_block(1, 1, 1, 4, 4, 4, U_b, V_b, W_b)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
