import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    # Build the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Fallback: Schoolbook decomposition (rank 112)
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Total rank: 49 (Strassen 4x4x4) + 44 (4 * 11 for 2x2x3)
    total_rank = 49 + 44
    
    U = [[0.0] * total_rank for _ in range(dim_u)]
    V = [[0.0] * total_rank for _ in range(dim_v)]
    W = [[0.0] * total_rank for _ in range(dim_w)]
    
    current_rank = 0
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    # Strassen 2x2x2 base terms: (u_coeffs, v_coeffs, w_coeffs)
    # u, v, w are dicts mapping (row, col) -> coeff for 2x2 matrices
    
    strassen_2x2 = [
        # M1: (A00+A11)*(B00+B11) -> C00 + C11
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (A10+A11)*B00 -> C10
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3: A00*(B01-B11) -> C01
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # M4: A11*(B10-B00) -> C00 + C10
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5: (A00+A01)*B11 -> C00 - C01
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (A10-A00)*(B00+B01) -> C11
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (A01-A11)*(B10+B11) -> C00
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # For 4x4x4: apply Strassen at block level (2x2 blocks of 2x2)
    # High-level Strassen terms for 4x4x4
    high_strassen = [
        # M1: (A00+A11)*(B00+B11) -> C00 + C11
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (A10+A11)*B00 -> C10
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M3: A00*(B01-B11) -> C01
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # M4: A11*(B10-B00) -> C00 + C10
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5: (A00+A01)*B11 -> C00 - C01
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (A10-A00)*(B00+B01) -> C11
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (A01-A11)*(B10+B11) -> C00
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for h_idx in range(7):
        u_h, v_h, w_h = high_strassen[h_idx]
        for l_idx in range(7):
            u_l, v_l, w_l = strassen_2x2[l_idx]
            
            r = current_rank
            current_rank += 1
            
            # Build U vector (4x4 = 16 entries)
            u_vec = [0.0] * 16
            for (bi, bj), ch in u_h.items():
                for (li, lj), cl in u_l.items():
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u_vec[i * 4 + j] += ch * cl
            
            # Build V vector (4x4 part of B, cols 0-3, 16 entries in 28-dim)
            v_vec = [0.0] * 28
            for (bi, bj), ch in v_h.items():
                for (li, lj), cl in v_l.items():
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    v_vec[i * 7 + j] += ch * cl
            
            # Build W vector (4x4 = 16 entries in 28-dim)
            w_vec = [0.0] * 28
            for (bi, bj), ch in w_h.items():
                for (li, lj), cl in w_l.items():
                    i = 2 * bi + li
                    k = 2 * bj + lj
                    w_vec[k * 4 + i] += ch * cl
            
            for idx in range(16):
                U[idx][r] = u_vec[idx]
            for idx in range(28):
                V[idx][r] = v_vec[idx]
            for idx in range(28):
                W[idx][r] = w_vec[idx]
            
            if time.time() > deadline:
                return get_schoolbook()
    
    # --- Part 2: 4x4x3 using 4 * (2x2x3) (Rank 44) ---
    # B columns 4,5,6. Split A into 2x2 blocks, B into 2x3 blocks.
    # For each A block (2x2) and corresponding B block (2x3), compute 2x2 result.
    # 2x2x3 rank 11: Strassen for cols 4,5 (rank 7) + schoolbook for col 6 (rank 4)
    
    # A blocks: (row_block, col_block) where row_block, col_block in {0,1}
    # B blocks: (row_block) where row_block in {0,1}, cols 4,5,6
    # C blocks: (row_block) where row_block in {0,1}, cols 4,5,6
    
    # Strassen 2x2x2 terms for the 2-column part
    strassen_2x2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for a_row_block in range(2):
        for a_col_block in range(2):
            b_row_block = a_col_block  # A_col_block matches B_row_block
            c_row_block = a_row_block  # A_row_block matches C_row_block
            
            # Strassen part: cols 4,5 (7 terms)
            for t_idx in range(7):
                u_sub, v_sub, w_sub = strassen_2x2x2[t_idx]
                
                r = current_rank
                current_rank += 1
                
                u_vec = [0.0] * 16
                for (li, lj), c in u_sub.items():
                    i = 2 * a_row_block + li
                    j = 2 * a_col_block + lj
                    u_vec[i * 4 + j] += c
                
                v_vec = [0.0] * 28
                for (li, lj), c in v_sub.items():
                    i = 2 * b_row_block + li
                    j = 4 + lj  # cols 4,5
                    v_vec[i * 7 + j] += c
                
                w_vec = [0.0] * 28
                for (li, lj), c in w_sub.items():
                    i = 2 * c_row_block + li
                    k = 4 + lj  # cols 4,5
                    w_vec[k * 4 + i] += c
                
                for idx in range(16):
                    U[idx][r] = u_vec[idx]
                for idx in range(28):
                    V[idx][r] = v_vec[idx]
                for idx in range(28):
                    W[idx][r] = w_vec[idx]
                
                if time.time() > deadline:
                    return get_schoolbook()
            
            # Schoolbook part: col 6 (4 terms)
            for i_local in range(2):
                for j_local in range(2):
                    r = current_rank
                    current_rank += 1
                    
                    i = 2 * a_row_block + i_local
                    j = 2 * a_col_block + j_local
                    b_row = 2 * b_row_block + j_local
                    c_row = 2 * c_row_block + i_local
                    
                    U[i * 4 + j][r] = 1.0
                    V[b_row * 7 + 6][r] = 1.0
                    W[6 * 4 + c_row][r] = 1.0
                    
                    if time.time() > deadline:
                        return get_schoolbook()
    
    # Verify the solution
    T_check = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for t in range(current_rank):
        U_t = np.array(U)[:, t]
        V_t = np.array(V)[:, t]
        W_t = np.array(W)[:, t]
        T_check += np.einsum('a,b,c->abc', U_t, V_t, W_t)
    
    if not np.allclose(T_check, T, atol=1e-10):
        return get_schoolbook()
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
