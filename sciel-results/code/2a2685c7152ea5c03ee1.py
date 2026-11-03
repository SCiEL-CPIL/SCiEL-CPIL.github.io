import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Partition indices
    # I: [0,1], [2]
    # J: [0,1], [2,3]
    # K: [0,1], [2,3], [4,5]
    
    I_parts = [[0, 1], [2]]
    J_parts = [[0, 1], [2, 3]]
    K_parts = [[0, 1], [2, 3], [4, 5]]
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, val):
        # u_idx, v_idx, w_idx are lists of (global_idx, coeff)
        # This function is a placeholder for logic, we will build columns directly
        pass

    for a_idx, I_part in enumerate(I_parts):
        for b_idx, J_part in enumerate(J_parts):
            for c_idx, K_part in enumerate(K_parts):
                n_a = len(I_part)
                m_b = len(J_part)
                p_c = len(K_part)
                
                if n_a == 2 and m_b == 2 and p_c == 2:
                    # Strassen
                    # 7 terms
                    # Local indices i,j,k in 0..1
                    # Global I = I_part[i], J = J_part[j], K = K_part[k]
                    
                    # Term 1: (a00+a11)(b00+b11) -> C00, C11
                    # U: a00, a11
                    # V: b00, b11
                    # W: C00, C11
                    
                    # We will create 7 columns
                    # Column 0
                    u_col = [0] * (n * m)
                    v_col = [0] * (m * p)
                    w_col = [0] * (p * n)
                    
                    # M1
                    # a00, a11
                    u_col[I_part[0] * m + J_part[0]] = 1
                    u_col[I_part[1] * m + J_part[1]] = 1
                    # b00, b11
                    v_col[J_part[0] * p + K_part[0]] = 1
                    v_col[J_part[1] * p + K_part[1]] = 1
                    # C00, C11
                    w_col[K_part[0] * n + I_part[0]] = 1
                    w_col[K_part[1] * n + I_part[1]] = 1
                    U.append(u_col); V.append(v_col); W.append(w_col)
                    
                    # M2: (a10+a11)b00 -> C10
                    u_col = [0] * (n * m); v_col = [0] * (m * p); w_col = [0] * (p * n)
                    u_col[I_part[1] * m + J_part[0]] = 1
                    u_col[I_part[1] * m + J_part[1]] = 1
                    v_col[J_part[0] * p + K_part[0]] = 1
                    w_col[K_part[0] * n + I_part[1]] = 1
                    U.append(u_col); V.append(v_col); W.append(w_col)
                    
                    # M3: a00(b01-b11) -> C01
                    u_col = [0] * (n * m); v_col = [0] * (m * p); w_col = [0] * (p * n)
                    u_col[I_part[0] * m + J_part[0]] = 1
                    v_col[J_part[0] * p + K_part[1]] = 1
                    v_col[J_part[1] * p + K_part[1]] = -1
                    w_col[K_part[1] * n + I_part[0]] = 1
                    U.append(u_col); V.append(v_col); W.append(w_col)
                    
                    # M4: a11(b10-b00) -> C00, C10
                    u_col = [0] * (n * m); v_col = [0] * (m * p); w_col = [0] * (p * n)
                    u_col[I_part[1] * m + J_part[1]] = 1
                    v_col[J_part[1] * p + K_part[0]] = 1
                    v_col[J_part[0] * p + K_part[0]] = -1
                    w_col[K_part[0] * n + I_part[0]] = 1
                    w_col[K_part[0] * n + I_part[1]] = 1
                    U.append(u_col); V.append(v_col); W.append(w_col)
                    
                    # M5: (a00+a01)b11 -> C00, C01
                    u_col = [0] * (n * m); v_col = [0] * (m * p); w_col = [0] * (p * n)
                    u_col[I_part[0] * m + J_part[0]] = 1
                    u_col[I_part[0] * m + J_part[1]] = 1
                    v_col[J_part[1] * p + K_part[1]] = 1
                    w_col[K_part[0] * n + I_part[0]] = -1
                    w_col[K_part[1] * n + I_part[0]] = 1
                    U.append(u_col); V.append(v_col); W.append(w_col)
                    
                    # M6: (a10-a00)(b00+b01) -> C11
                    u_col = [0] * (n * m); v_col = [0] * (m * p); w_col = [0] * (p * n)
                    u_col[I_part[1] * m + J_part[0]] = 1
                    u_col[I_part[0] * m + J_part[0]] = -1
                    v_col[J_part[0] * p + K_part[0]] = 1
                    v_col[J_part[0] * p + K_part[1]] = 1
                    w_col[K_part[1] * n + I_part[1]] = 1
                    U.append(u_col); V.append(v_col); W.append(w_col)
                    
                    # M7: (a01-a11)(b10+b11) -> C00, C11
                    u_col = [0] * (n * m); v_col = [0] * (m * p); w_col = [0] * (p * n)
                    u_col[I_part[0] * m + J_part[1]] = 1
                    u_col[I_part[1] * m + J_part[1]] = -1
                    v_col[J_part[1] * p + K_part[0]] = 1
                    v_col[J_part[1] * p + K_part[1]] = 1
                    w_col[K_part[0] * n + I_part[0]] = 1
                    w_col[K_part[1] * n + I_part[1]] = 1
                    U.append(u_col); V.append(v_col); W.append(w_col)
                    
                else:
                    # Schoolbook
                    for i_local in range(n_a):
                        for j_local in range(m_b):
                            for k_local in range(p_c):
                                u_col = [0] * (n * m)
                                v_col = [0] * (m * p)
                                w_col = [0] * (p * n)
                                
                                i_g = I_part[i_local]
                                j_g = J_part[j_local]
                                k_g = K_part[k_local]
                                
                                u_col[i_g * m + j_g] = 1
                                v_col[j_g * p + k_g] = 1
                                w_col[k_g * n + i_g] = 1
                                
                                U.append(u_col)
                                V.append(v_col)
                                W.append(w_col)
                                
                                if time.time() > deadline:
                                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
