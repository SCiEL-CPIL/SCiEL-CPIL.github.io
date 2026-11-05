import time
import numpy as np

DIMENSIONS = [6, 6, 6]

_DEADLINE = time.time() + 140

def construct():
    start_time = time.time()
    
    n, m, p = 6, 6, 6
    
    def _schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _strassen_2x2_schoolbook_3x3():
        block_size = 3
        num_blocks = 2
        dim = n
        
        rank_inner = 27
        rank_outer = 7
        total_rank = rank_inner * rank_outer
        
        U = np.zeros((dim * dim, total_rank))
        V = np.zeros((dim * dim, total_rank))
        W = np.zeros((dim * dim, total_rank))
        
        col = 0
        
        strassen_steps = [
            {'A': { (0,0): 1, (1,1): 1 }, 'B': { (0,0): 1, (1,1): 1 }, 'C': { (0,0): 1, (1,1): 1 }},
            {'A': { (1,0): 1, (1,1): 1 }, 'B': { (0,0): 1 }, 'C': { (1,0): 1, (1,1): -1 }},
            {'A': { (0,0): 1 }, 'B': { (0,1): 1, (1,1): -1 }, 'C': { (0,1): 1, (1,1): 1 }},
            {'A': { (1,1): 1 }, 'B': { (1,0): 1, (0,0): -1 }, 'C': { (0,0): 1, (1,0): 1 }},
            {'A': { (0,0): 1, (0,1): 1 }, 'B': { (1,1): 1 }, 'C': { (0,0): -1, (0,1): 1 }},
            {'A': { (1,0): 1, (0,0): -1 }, 'B': { (0,0): 1, (0,1): 1 }, 'C': { (1,1): 1 }},
            {'A': { (0,1): 1, (1,1): -1 }, 'B': { (1,0): 1, (1,1): 1 }, 'C': { (0,0): 1 }}
        ]
        
        schoolbook_steps = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    schoolbook_steps.append({
                        'A': { (i,j): 1 },
                        'B': { (j,k): 1 },
                        'C': { (i,k): 1 }
                    })
        
        for s_step in strassen_steps:
            A_coeffs_outer = s_step['A']
            B_coeffs_outer = s_step['B']
            C_coeffs_outer = s_step['C']
            
            for sb_step in schoolbook_steps:
                A_coeffs_inner = sb_step['A']
                B_coeffs_inner = sb_step['B']
                C_coeffs_inner = sb_step['C']
                
                for (I, J), ca_out in A_coeffs_outer.items():
                    for (u, v), ca_in in A_coeffs_inner.items():
                        i_global = I * block_size + u
                        j_global = J * block_size + v
                        idx_u = i_global * dim + j_global
                        U[idx_u, col] += ca_out * ca_in
                
                for (J, K), cb_out in B_coeffs_outer.items():
                    for (v, w), cb_in in B_coeffs_inner.items():
                        j_global = J * block_size + v
                        k_global = K * block_size + w
                        idx_v = j_global * dim + k_global
                        V[idx_v, col] += cb_out * cb_in
                
                for (I, K), cc_out in C_coeffs_outer.items():
                    for (u, w), cc_in in C_coeffs_inner.items():
                        i_global = I * block_size + u
                        k_global = K * block_size + w
                        idx_w = k_global * dim + i_global
                        W[idx_w, col] += cc_out * cc_in
                
                col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _strassen_2x2_laderman_3x3():
        block_size = 3
        num_blocks = 2
        dim = n
        
        rank_inner = 23
        rank_outer = 7
        total_rank = rank_inner * rank_outer
        
        U = np.zeros((dim * dim, total_rank))
        V = np.zeros((dim * dim, total_rank))
        W = np.zeros((dim * dim, total_rank))
        
        col = 0
        
        strassen_steps = [
            {'A': { (0,0): 1, (1,1): 1 }, 'B': { (0,0): 1, (1,1): 1 }, 'C': { (0,0): 1, (1,1): 1 }},
            {'A': { (1,0): 1, (1,1): 1 }, 'B': { (0,0): 1 }, 'C': { (1,0): 1, (1,1): -1 }},
            {'A': { (0,0): 1 }, 'B': { (0,1): 1, (1,1): -1 }, 'C': { (0,1): 1, (1,1): 1 }},
            {'A': { (1,1): 1 }, 'B': { (1,0): 1, (0,0): -1 }, 'C': { (0,0): 1, (1,0): 1 }},
            {'A': { (0,0): 1, (0,1): 1 }, 'B': { (1,1): 1 }, 'C': { (0,0): -1, (0,1): 1 }},
            {'A': { (1,0): 1, (0,0): -1 }, 'B': { (0,0): 1, (0,1): 1 }, 'C': { (1,1): 1 }},
            {'A': { (0,1): 1, (1,1): -1 }, 'B': { (1,0): 1, (1,1): 1 }, 'C': { (0,0): 1 }}
        ]
        
        # Laderman algorithm for 3x3 matrix multiplication (Rank 23)
        laderman_steps = [
            ({(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (1,1):1, (2,2):1}, {(0,0):1, (0,1):1, (0,2):1, (1,0):1, (1,1):1, (1,2):1, (2,0):1, (2,1):1, (2,2):1}),
            ({(0,0):1, (1,1):1}, {(1,1):1, (2,2):1}, {(0,1):1, (0,2):1, (1,1):1, (1,2):1}),
            ({(0,0):1, (2,2):1}, {(2,2):1, (0,0):1}, {(0,2):1, (0,0):1, (2,2):1, (2,0):1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
            ({(1,1):1}, {(1,2):1, (2,2):-1}, {(1,2):1}),
            ({(2,2):1}, {(2,0):1, (0,0):-1}, {(2,0):1}),
            ({(0,1):1, (1,1):1}, {(1,0):1, (2,0):1}, {(0,0):1, (1,0):1}),
            ({(1,2):1, (2,2):1}, {(2,1):1, (0,1):1}, {(1,1):1, (2,1):1}),
            ({(2,0):1, (0,0):1}, {(0,2):1, (1,2):1}, {(2,2):1, (0,2):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1, (1,0):-1}, {(0,1):1}),
            ({(1,1):1, (1,2):1}, {(2,2):1, (2,1):-1}, {(1,2):1}),
            ({(2,2):1, (2,0):1}, {(0,0):1, (0,2):-1}, {(2,0):1}),
            ({(0,1):1, (1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1}),
            ({(1,2):1, (2,2):1}, {(2,1):1, (1,1):-1}, {(1,1):1}),
            ({(2,0):1, (0,0):1}, {(0,2):1, (2,2):-1}, {(2,2):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1, (2,1):-1}, {(0,1):1}),
            ({(1,1):1, (1,2):1}, {(2,2):1, (0,2):-1}, {(1,2):1}),
            ({(2,2):1, (2,0):1}, {(0,0):1, (1,0):-1}, {(2,0):1}),
            ({(0,0):1, (0,1):-1, (1,1):1}, {(1,1):1, (2,1):1, (1,0):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1, (1,2):-1, (2,2):1}, {(2,2):1, (0,2):1, (2,1):-1}, {(1,2):1, (2,2):1}),
            ({(2,2):1, (2,0):-1, (0,0):1}, {(0,0):1, (1,0):1, (0,2):-1}, {(2,0):1, (0,0):1}),
            ({(0,1):1, (1,1):1, (1,2):-1}, {(1,2):1, (2,2):-1, (2,1):1}, {(0,2):1, (1,1):1}),
            ({(1,2):1, (2,2):1, (2,0):-1}, {(2,0):1, (0,0):-1, (0,2):1}, {(1,0):1, (2,2):1})
        ]
        
        for s_step in strassen_steps:
            A_coeffs_outer = s_step['A']
            B_coeffs_outer = s_step['B']
            C_coeffs_outer = s_step['C']
            
            for l_step in laderman_steps:
                A_coeffs_inner = l_step[0]
                B_coeffs_inner = l_step[1]
                C_coeffs_inner = l_step[2]
                
                for (I, J), ca_out in A_coeffs_outer.items():
                    for (u, v), ca_in in A_coeffs_inner.items():
                        i_global = I * block_size + u
                        j_global = J * block_size + v
                        idx_u = i_global * dim + j_global
                        U[idx_u, col] += ca_out * ca_in
                
                for (J, K), cb_out in B_coeffs_outer.items():
                    for (v, w), cb_in in B_coeffs_inner.items():
                        j_global = J * block_size + v
                        k_global = K * block_size + w
                        idx_v = j_global * dim + k_global
                        V[idx_v, col] += cb_out * cb_in
                
                for (I, K), cc_out in C_coeffs_outer.items():
                    for (u, w), cc_in in C_coeffs_inner.items():
                        i_global = I * block_size + u
                        k_global = K * block_size + w
                        idx_w = k_global * dim + i_global
                        W[idx_w, col] += cc_out * cc_in
                
                col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _verify(U_list, V_list, W_list):
        U_np = np.array(U_list, dtype=float)
        V_np = np.array(V_list, dtype=float)
        W_np = np.array(W_list, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_np[i * m + j, :]
                for k in range(p):
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    s = np.dot(u_row, v_row * w_row)
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < 1e-10

    try:
        # Try Laderman first
        if time.time() < _DEADLINE:
            res_laderman = _strassen_2x2_laderman_3x3()
            if _verify(res_laderman["U"], res_laderman["V"], res_laderman["W"]):
                return res_laderman
        
        # Fallback to Strassen-Schoolbook
        if time.time() < _DEADLINE:
            res_strassen = _strassen_2x2_schoolbook_3x3()
            if _verify(res_strassen["U"], res_strassen["V"], res_strassen["W"]):
                return res_strassen
                
    except Exception:
        pass
        
    return _schoolbook()
