"""Combined program: Strassen-based structural decomposition for 4x4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 100

    def build_decomposition():
        total_rank = 93
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        current_rank = 0
        
        strassen_2x2 = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
        ]
        
        strassen_4x4_outer = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}),
            ({0: 1}, {1: 1, 3: -1}),
            ({3: 1}, {2: 1, 0: -1}),
            ({0: 1, 1: 1}, {3: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}),
        ]
        
        c_blocks_4x4 = [
            {(0,0): 1.0, (1,1): 1.0},
            {(1,0): 1.0, (1,1): -1.0},
            {(0,1): 1.0, (1,1): 1.0},
            {(0,0): 1.0, (1,0): 1.0},
            {(0,0): -1.0, (0,1): 1.0},
            {(1,1): 1.0},
            {(0,0): 1.0},
        ]
        
        for m_idx in range(7):
            x_def, y_def = strassen_4x4_outer[m_idx]
            c_block = c_blocks_4x4[m_idx]
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                
                for (li, lj), cx in x_sub.items():
                    for b, cb in x_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        U[i * m + j, r] += cx * cb
                
                for (li, lj), cy in y_sub.items():
                    for b, cb in y_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        V[i * p + j, r] += cy * cb
                
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    for (br, bc), cb_c in c_block.items():
                        if cb_c == 0:
                            continue
                        i_final = 2 * br + li
                        k_final = 2 * bc + lj
                        W[k_final * n + i_final, r] += cc * cb_c
        
        def add_strassen_2x2(A_block_rows, A_block_cols, B_block_rows, B_block_cols, C_block_rows, C_block_cols):
            nonlocal current_rank
            for t_idx in range(7):
                r = current_rank
                current_rank += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                
                for (li, lj), cx in x_sub.items():
                    i = A_block_rows[li]
                    j = A_block_cols[lj]
                    U[i * m + j, r] += cx
                    
                for (li, lj), cy in y_sub.items():
                    i = B_block_rows[li]
                    j = B_block_cols[lj]
                    V[i * p + j, r] += cy
                    
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    i_final = C_block_rows[li]
                    k_final = C_block_cols[lj]
                    W[k_final * n + i_final, r] += cc

        A00_r = [0, 1]; A00_c = [0, 1]
        A01_r = [0, 1]; A01_c = [2, 3]
        A10_r = [2, 3]; A10_c = [0, 1]
        A11_r = [2, 3]; A11_c = [2, 3]
        
        B_top_r = [0, 1]; B_top_c = [4, 5]
        B_bot_r = [2, 3]; B_bot_c = [4, 5]
        
        C_top_r = [0, 1]; C_top_c = [4, 5]
        C_bot_r = [2, 3]; C_bot_c = [4, 5]
        
        add_strassen_2x2(A00_r, A00_c, B_top_r, B_top_c, C_top_r, C_top_c)
        add_strassen_2x2(A01_r, A01_c, B_bot_r, B_bot_c, C_top_r, C_top_c)
        add_strassen_2x2(A10_r, A10_c, B_top_r, B_top_c, C_bot_r, C_bot_c)
        add_strassen_2x2(A11_r, A11_c, B_bot_r, B_bot_c, C_bot_r, C_bot_c)

        for i in range(n):
            for j in range(m):
                r = current_rank
                current_rank += 1
                U[i * m + j, r] = 1.0
                V[j * p + 6, r] = 1.0
                W[6 * n + i, r] = 1.0
        
        assert current_rank == total_rank, f"Expected {total_rank}, got {current_rank}"
        
        return U.tolist(), V.tolist(), W.tolist()
    
    def verify(U, V, W, r):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))
    
    U, V, W = build_decomposition()
    rank = len(U[0])
    err = verify(U, V, W, rank)
    
    if err < 1e-10:
        return {"U": U, "V": V, "W": W}
    
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
