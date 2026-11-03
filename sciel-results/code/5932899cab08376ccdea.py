def construct():
    n, m, p = DIMENSIONS # 2, 4, 7
    # Rank 50 decomposition
    # 2 blocks of (2x2x7)
    # Each (2x2x7) is 3x(2x2x2) + 1x(2x2x1)
    # Total rank = 2 * (3*7 + 4) = 2 * 25 = 50
    
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    for block in range(2):
        # block 0: A cols 0,1, B rows 0,1
        # block 1: A cols 2,3, B rows 2,3
        j_a_offset = block * 2
        j_b_offset = block * 2
        
        # Part 1: 2x2x2 (B cols 0,1)
        U_s, V_s, W_s = _get_strassen_222()
        r_s = U_s.shape[1]
        for r in range(r_s):
            for i_s in range(2):
                for j_s in range(2):
                    # U
                    # U_sub[i_s*2 + j_s, r]
                    val_u = U_s[i_s * 2 + j_s, r]
                    if val_u != 0:
                        i_g = i_s
                        j_g = j_a_offset + j_s
                        U[i_g * m + j_g, col_offset + r] = val_u
                        
                        # V
                        # V_sub[j_s*2 + k_s, r]
                        # Note: V_sub index is j_s * sub_p + k_s. sub_p=2.
                        # We need to iterate k_s as well?
                        # Actually, the sub-decomposition is defined for all i_s, j_s, k_s.
                        # It's better to iterate over all non-zero entries in U_s, V_s, W_s.
                        
        # This manual iteration is error-prone. 
        # Let's just copy the whole sub-matrices with index mapping.
        
        # Mapping for Part 1 (2x2x2, k_offset=0)
        for r in range(r_s):
            for i_s in range(2):
                for j_s in range(2):
                    u_val = U_s[i_s * 2 + j_s, r]
                    if u_val != 0:
                        U[i_s * m + (j_a_offset + j_s), col_offset + r] += u_val
                        
            for j_s in range(2):
                for k_s in range(2):
                    v_val = V_s[j_s * 2 + k_s, r]
                    if v_val != 0:
                        V[(j_b_offset + j_s) * p + (0 + k_s), col_offset + r] += v_val
                        
            for k_s in range(2):
                for i_s in range(2):
                    w_val = W_s[k_s * 2 + i_s, r]
                    if w_val != 0:
                        W[(0 + k_s) * n + i_s, col_offset + r] += w_val
                        
        col_offset += r_s
        
        # Part 2: 2x2x2 (B cols 2,3)
        U_s, V_s, W_s = _get_strassen_222()
        r_s = U_s.shape[1]
        k_offset = 2
        for r in range(r_s):
            for i_s in range(2):
                for j_s in range(2):
                    u_val = U_s[i_s * 2 + j_s, r]
                    if u_val != 0:
                        U[i_s * m + (j_a_offset + j_s), col_offset + r] += u_val
                        
            for j_s in range(2):
                for k_s in range(2):
                    v_val = V_s[j_s * 2 + k_s, r]
                    if v_val != 0:
                        V[(j_b_offset + j_s) * p + (k_offset + k_s), col_offset + r] += v_val
                        
            for k_s in range(2):
                for i_s in range(2):
                    w_val = W_s[k_s * 2 + i_s, r]
                    if w_val != 0:
                        W[(k_offset + k_s) * n + i_s, col_offset + r] += w_val
                        
        col_offset += r_s
        
        # Part 3: 2x2x2 (B cols 4,5)
        U_s, V_s, W_s = _get_strassen_222()
        r_s = U_s.shape[1]
        k_offset = 4
        for r in range(r_s):
            for i_s in range(2):
                for j_s in range(2):
                    u_val = U_s[i_s * 2 + j_s, r]
                    if u_val != 0:
                        U[i_s * m + (j_a_offset + j_s), col_offset + r] += u_val
                        
            for j_s in range(2):
                for k_s in range(2):
                    v_val = V_s[j_s * 2 + k_s, r]
                    if v_val != 0:
                        V[(j_b_offset + j_s) * p + (k_offset + k_s), col_offset + r] += v_val
                        
            for k_s in range(2):
                for i_s in range(2):
                    w_val = W_s[k_s * 2 + i_s, r]
                    if w_val != 0:
                        W[(k_offset + k_s) * n + i_s, col_offset + r] += w_val
                        
        col_offset += r_s
        
        # Part 4: 2x2x1 (B col 6)
        U_s, V_s, W_s = _get_school_221()
        r_s = U_s.shape[1]
        k_offset = 6
        for r in range(r_s):
            for i_s in range(2):
                for j_s in range(2):
                    u_val = U_s[i_s * 2 + j_s, r]
                    if u_val != 0:
                        U[i_s * m + (j_a_offset + j_s), col_offset + r] += u_val
                        
            for j_s in range(2):
                # k_s is always 0 for 2x2x1
                v_val = V_s[j_s * 1 + 0, r]
                if v_val != 0:
                    V[(j_b_offset + j_s) * p + (k_offset + 0), col_offset + r] += v_val
                    
            # k_s is always 0
            for i_s in range(2):
                w_val = W_s[0 * 2 + i_s, r]
                if w_val != 0:
                    W[(k_offset + 0) * n + i_s, col_offset + r] += w_val
                    
        col_offset += r_s
        
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        r = n * m * p
        U_fb = np.zeros((n * m, r))
        V_fb = np.zeros((m * p, r))
        W_fb = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
