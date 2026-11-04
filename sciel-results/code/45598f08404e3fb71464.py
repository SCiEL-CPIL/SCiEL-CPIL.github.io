import numpy as np
import time

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    # Total rank: 49 (4x4x4 Strassen) + 44 (4 * 2x2x3, each rank 11)
    total_rank = 49 + 44
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # --- Part 1: 4x4x4 using two-level Strassen (Rank 49) ---
    # Outer Strassen: 7 terms, each combining 2x2 blocks
    # Inner Strassen: 7 terms per outer term
    
    # Outer Strassen terms: (x_block_def, y_block_def)
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    outer_strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}),          # M2
        ({0: 1}, {1: 1, 3: -1}),         # M3
        ({3: 1}, {2: 1, 0: -1}),         # M4
        ({0: 1, 1: 1}, {3: 1}),          # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),   # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1})    # M7
    ]
    
    # Inner Strassen terms for 2x2x2: (x_sub, y_sub) with local coords
    inner_strassen = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),   # p0
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),              # p1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),             # p2
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),             # p3
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),              # p4
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),   # p5
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})    # p6
    ]
    
    # C block coefficients for inner Strassen output
    c_inner = {
        0: {(0,0): 1, (1,1): 1},
        1: {(1,0): 1, (1,1): -1},
        2: {(0,1): 1, (1,1): 1},
        3: {(0,0): 1, (1,0): 1},
        4: {(0,0): -1, (0,1): 1},
        5: {(1,1): 1},
        6: {(0,0): 1}
    }
    
    # C block coefficients for outer Strassen output
    c_outer = {
        0: {(0,0): 1, (1,1): 1},
        1: {(1,0): 1, (1,1): -1},
        2: {(0,1): 1, (1,1): 1},
        3: {(0,0): 1, (1,0): 1},
        4: {(0,0): -1, (0,1): 1},
        5: {(1,1): 1},
        6: {(0,0): 1}
    }
    
    for m_idx in range(7):
        x_def, y_def = outer_strassen[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub = inner_strassen[t_idx]
            
            u_coeffs = [0.0] * (n * m)
            v_coeffs = [0.0] * (m * p)
            
            # Build U coefficients
            for (li, lj), c_x in x_sub.items():
                for b, coeff in x_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    u_idx = i * m + j
                    u_coeffs[u_idx] += c_x * coeff
            
            # Build V coefficients
            for (li, lj), c_y in y_sub.items():
                for b, coeff in y_def.items():
                    bi = b // 2
                    bj = b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    v_idx = i * p + j
                    v_coeffs[v_idx] += c_y * coeff
            
            for i in range(n * m):
                U[i][r] = u_coeffs[i]
            for j in range(m * p):
                V[j][r] = v_coeffs[j]
            
            # Build W coefficients
            w_coeffs = [0.0] * (p * n)
            for (li, lj), cm_c in c_inner[t_idx].items():
                for (br, bc), cb_c in c_outer[m_idx].items():
                    if cm_c == 0 or cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    w_idx = k_final * n + i_final
                    w_coeffs[w_idx] += cm_c * cb_c
            
            for i in range(p * n):
                W[i][r] = w_coeffs[i]
    
    # --- Part 2: 4x4x3 using 4 * (2x2x3) (Rank 44) ---
    # Columns 4,5,6 of B
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B top rows: 0,1; B bot rows: 2,3
    
    def add_2x2x3(a_rb, a_cb, b_rb, c_rb):
        nonlocal current_rank
        
        # 7 Strassen terms for cols 4,5
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub = inner_strassen[t_idx]
            
            u_coeffs = [0.0] * (n * m)
            v_coeffs = [0.0] * (m * p)
            
            for (li, lj), c in x_sub.items():
                i = 2 * a_rb + li
                j = 2 * a_cb + lj
                u_idx = i * m + j
                u_coeffs[u_idx] += c
            
            for (li, lj), c in y_sub.items():
                i = 2 * b_rb + li
                j = 4 + lj  # Cols 4,5
                v_idx = i * p + j
                v_coeffs[v_idx] += c
            
            for i in range(n * m):
                U[i][r] = u_coeffs[i]
            for j in range(m * p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p * n)
            for (li, lj), c in c_inner[t_idx].items():
                i_final = 2 * c_rb + li
                k_final = 4 + lj
                w_idx = k_final * n + i_final
                w_coeffs[w_idx] += c
            
            for i in range(p * n):
                W[i][r] = w_coeffs[i]
        
        # 4 schoolbook terms for col 6
        schoolbook = [
            ({(0,0): 1}, {(0,0): 1}, (0,0)),
            ({(0,1): 1}, {(1,0): 1}, (0,0)),
            ({(1,0): 1}, {(0,0): 1}, (1,0)),
            ({(1,1): 1}, {(1,0): 1}, (1,0))
        ]
        
        for t_idx in range(4):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_pos = schoolbook[t_idx]
            
            u_coeffs = [0.0] * (n * m)
            v_coeffs = [0.0] * (m * p)
            
            for (li, lj), c in x_sub.items():
                i = 2 * a_rb + li
                j = 2 * a_cb + lj
                u_idx = i * m + j
                u_coeffs[u_idx] += c
            
            for (li, lj), c in y_sub.items():
                i = 2 * b_rb + li
                j = 6  # Col 6
                v_idx = i * p + j
                v_coeffs[v_idx] += c
            
            for i in range(n * m):
                U[i][r] = u_coeffs[i]
            for j in range(m * p):
                V[j][r] = v_coeffs[j]
            
            w_coeffs = [0.0] * (p * n)
            i_final = 2 * c_rb + c_pos[0]
            k_final = 6
            w_idx = k_final * n + i_final
            w_coeffs[w_idx] = 1.0
            
            for i in range(p * n):
                W[i][r] = w_coeffs[i]
    
    # C[0:2, 4:7] = A[0:2,0:2]*B[0:2,4:7] + A[0:2,2:4]*B[2:4,4:7]
    add_2x2x3(0, 0, 0, 0)  # A_00 * B_top -> C_top
    add_2x2x3(0, 1, 1, 0)  # A_01 * B_bot -> C_top
    add_2x2x3(1, 0, 0, 1)  # A_10 * B_top -> C_bot
    add_2x2x3(1, 1, 1, 1)  # A_11 * B_bot -> C_bot
    
    # Trim to actual rank used
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    # Quick verification
    if _time_remaining() > 10:
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        import random
        random.seed(123)
        errors = 0
        for _ in range(200):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            val = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
            if abs(val - 1.0) > 1e-6:
                errors += 1
        
        if errors > 0:
            # Fallback: use schoolbook if verification fails
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
