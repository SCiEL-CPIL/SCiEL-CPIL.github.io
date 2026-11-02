        if nc >= 2 and mc >= 2 and pc >= 2:
            cols.extend(strassen_2x2x2_cols(ii, jj, kk, n_dim, m_dim, p_dim))
            
            # Recurse on remainders
            # 1. (nc-2) x mc x pc at (ii+2, jj, kk)
            if nc > 2:
                decompose_recursive(nc - 2, mc, pc, ii + 2, jj, kk, n_dim, m_dim, p_dim, cols)
            
            # 2. 2 x (mc-2) x pc at (ii, jj+2, kk)
            if mc > 2:
                decompose_recursive(2, mc - 2, pc, ii, jj + 2, kk, n_dim, m_dim, p_dim, cols)
            
            # 3. 2 x 2 x (pc-2) at (ii, jj, kk+2)
            if pc > 2:
                decompose_recursive(2, 2, pc - 2, ii, jj, kk + 2, n_dim, m_dim, p_dim, cols)
            
            return
