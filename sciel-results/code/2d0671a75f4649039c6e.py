import time

_START_TIME = time.time()
TIME_LIMIT = 140.0

def construct():
    n, m, p = 2, 3, 8
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        for idx, c in zip(u_idx, u_coeff):
            if c != 0:
                U[idx][col] += c
        for idx, c in zip(v_idx, v_coeff):
            if c != 0:
                V[idx][col] += c
        for idx, c in zip(w_idx, w_coeff):
            if c != 0:
                W[idx][col] += c

    current_col = 0
    
    # Part 1: M_{2,2,8} using 4 blocks of M_{2,2,2} (Strassen)
    # A indices: 0:(0,0), 1:(0,1), 3:(1,0), 4:(1,1)
    # B indices for block k: 
    #   B_{0, 2k}, B_{0, 2k+1} -> idx 0*p + 2k, 0*p + 2k+1
    #   B_{1, 2k}, B_{1, 2k+1} -> idx 1*p + 2k, 1*p + 2k+1
    # C indices for block k:
    #   C_{0, 2k}, C_{0, 2k+1} -> idx (2k)*n + 0, (2k+1)*n + 0
    #   C_{1, 2k}, C_{1, 2k+1} -> idx (2k)*n + 1, (2k+1)*n + 1
    
    a_idx, b_idx, c_idx, d_idx = 0, 1, 3, 4
    
    for k_block in range(4):
        if time.time() - _START_TIME > TIME_LIMIT:
            break
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        
        e_idx = 0 * p + k0
        f_idx = 0 * p + k1
        g_idx = 1 * p + k0
        h_idx = 1 * p + k1
        
        w_00 = k0 * n
        w_01 = k1 * n
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # Strassen terms for 2x2x2
        add_term(current_col, [a_idx, d_idx], [1, 1], [e_idx, h_idx], [1, 1], [w_00, w_11], [1, 1])
        current_col += 1
        add_term(current_col, [c_idx, d_idx], [1, 1], [e_idx], [1], [w_10, w_11], [1, -1])
        current_col += 1
        add_term(current_col, [a_idx], [1], [f_idx, h_idx], [1, -1], [w_01, w_11], [1, 1])
        current_col += 1
        add_term(current_col, [d_idx], [1], [g_idx, e_idx], [1, -1], [w_00, w_10], [1, 1])
        current_col += 1
        add_term(current_col, [a_idx, b_idx], [1, 1], [h_idx], [1], [w_00, w_01], [-1, 1])
        current_col += 1
        add_term(current_col, [c_idx, a_idx], [1, -1], [e_idx, f_idx], [1, 1], [w_11], [1])
        current_col += 1
        add_term(current_col, [b_idx, d_idx], [1, -1], [g_idx, h_idx], [1, 1], [w_00], [1])
        current_col += 1

    # Part 2: M_{2,1,8} using rank 2
    # A indices: 2:(0,2), 5:(1,2)
    # B indices: 2*p + k for k=0..7
    # C indices: k*n + 0, k*n + 1
    
    a2_idx = 2
    a3_idx = 5
    
    # Term 1: A_{0,2} * B_{2,:} -> C_{0,:}
    v_indices = [2 * p + k for k in range(p)]
    v_coeffs = [1] * p
    w_indices = [k * n + 0 for k in range(p)]
    w_coeffs = [1] * p
    add_term(current_col, [a2_idx], [1], v_indices, v_coeffs, w_indices, w_coeffs)
    current_col += 1
    
    # Term 2: A_{1,2} * B_{2,:} -> C_{1,:}
    w_indices = [k * n + 1 for k in range(p)]
    w_coeffs = [1] * p
    add_term(current_col, [a3_idx], [1], v_indices, v_coeffs, w_indices, w_coeffs)
    current_col += 1

    return {"U": U, "V": V, "W": W}
