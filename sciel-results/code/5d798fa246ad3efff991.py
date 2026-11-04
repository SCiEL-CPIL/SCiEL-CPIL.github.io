def _schoolbook_block(U, V, W, n_block, m_block, p_block, start_rank, 
                      u_row_offset, u_col_offset, 
                      v_row_offset, v_col_offset, 
                      w_row_offset, w_col_offset, global_n=5):
    rank = start_rank
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                u_idx = (u_row_offset + i) * 5 + (u_col_offset + j) # Global U size 5x5
                v_idx = (v_row_offset + j) * 5 + (v_col_offset + k) # Global V size 5x5
                w_idx = (w_col_offset + k) * global_n + (w_row_offset + i) # Global W index k*5+i
                
                U[u_idx][rank] = 1
                V[v_idx][rank] = 1
                W[w_idx][rank] = 1
                rank += 1
    return rank
