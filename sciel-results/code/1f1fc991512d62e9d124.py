    def build_4x4x6():
        """Rank-77 Strassen decomposition for 4x4 * 4x6 -> 4x6."""
        n, m, p = 4, 4, 6
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = len(U_sub[0])
        total_rank = 7 * rank_sub
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        terms = [
            {'A': [(0,1),(3,1)], 'B': [(0,1),(3,1)], 'C': [(0,1),(3,1)]},
            {'A': [(2,1),(3,1)], 'B': [(0,1)], 'C': [(2,1),(3,-1)]},
            {'A': [(0,1)], 'B': [(1,1),(3,-1)], 'C': [(1,1),(3,1)]},
            {'A': [(3,1)], 'B': [(2,1),(0,-1)], 'C': [(0,1),(2,1)]},
            {'A': [(0,1),(1,1)], 'B': [(3,1)], 'C': [(0,-1),(1,1)]},
            {'A': [(2,1),(0,-1)], 'B': [(0,1),(1,1)], 'C': [(3,1)]},
            {'A': [(1,1),(3,-1)], 'B': [(2,1),(3,1)], 'C': [(0,1)]},
        ]
        u_offset = {0:(0,0), 1:(0,2), 2:(2,0), 3:(2,2)}
        v_offset = {0:(0,0), 1:(0,3), 2:(2,0), 3:(2,3)}
        w_offset = {0:(0,0), 1:(3,0), 2:(0,2), 3:(3,2)}
        
        U_sub_arr = np.array(U_sub)
        V_sub_arr = np.array(V_sub)
        W_sub_arr = np.array(W_sub)
        
        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub
            # Vectorized addition for U
            for u_local in range(4):
                u_val = U_sub_arr[u_local, :]
                if not np.any(u_val): continue
                i_local = u_local // 2; j_local = u_local % 2
                for block, coeff in term['A']:
                    ro, co = u_offset[block]
                    g_u = (i_local + ro) * m + (j_local + co)
                    U[g_u, base_col:base_col+rank_sub] += u_val * coeff
            # Vectorized addition for V
            for v_local in range(6):
                v_val = V_sub_arr[v_local, :]
                if not np.any(v_val): continue
                j_local = v_local // 3; k_local = v_local % 3
                for block, coeff in term['B']:
                    ro, co = v_offset[block]
                    g_v = (j_local + ro) * p + (k_local + co)
                    V[g_v, base_col:base_col+rank_sub] += v_val * coeff
            # Vectorized addition for W
            for w_local in range(6):
                w_val = W_sub_arr[w_local, :]
                if not np.any(w_val): continue
                k_local = w_local // 2; i_local = w_local % 2
                for block, coeff in term['C']:
                    ro, co = w_offset[block]
                    g_w = (k_local + ro) * n + (i_local + co)
                    W[g_w, base_col:base_col+rank_sub] += w_val * coeff
        return U.tolist(), V.tolist(), W.tolist()
