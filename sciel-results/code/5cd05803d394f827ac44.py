def construct():
    n, m, p = DIMENSIONS
    I_parts = [[0, 1], [2, 3], [4]]
    J_parts = [[0, 1], [2, 3], [4]]
    K_parts = [[0, 1], [2, 3], [4]]
    
    total_rank = 117
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_col = 0
    
    for a in range(3):
        for b in range(3):
            for c in range(3):
                I = I_parts[a]
                J = J_parts[b]
                K = K_parts[c]
                n_loc = len(I)
                m_loc = len(J)
                p_loc = len(K)
                
                if n_loc == 2 and m_loc == 2 and p_loc == 2:
                    U_loc, V_loc, W_loc = strassen_222()
                    rank_loc = 7
                else:
                    U_loc, V_loc, W_loc = schoolbook(n_loc, m_loc, p_loc)
                    rank_loc = n_loc * m_loc * p_loc
                
                for r in range(rank_loc):
                    # U
                    for i_loc in range(n_loc):
                        for j_loc in range(m_loc):
                            val = U_loc[i_loc*m_loc + j_loc][r]
                            if val != 0:
                                i_glob = I[i_loc]
                                j_glob = J[j_loc]
                                U[i_glob*m + j_glob][current_col + r] = val
                    # V
                    for j_loc in range(m_loc):
                        for k_loc in range(p_loc):
                            val = V_loc[j_loc*p_loc + k_loc][r]
                            if val != 0:
                                j_glob = J[j_loc]
                                k_glob = K[k_loc]
                                V[j_glob*p + k_glob][current_col + r] = val
                    # W
                    for k_loc in range(p_loc):
                        for i_loc in range(n_loc):
                            val = W_loc[k_loc*n_loc + i_loc][r]
                            if val != 0:
                                k_glob = K[k_loc]
                                i_glob = I[i_loc]
                                W[k_glob*n + i_glob][current_col + r] = val
                                
                current_col += rank_loc
                
    return {"U": U, "V": V, "W": W}
