    def add_schoolbook_1mp(i_off, j_off, k_off, m_dim, p_dim):
        # A is 1 x m_dim
        # B is m_dim x p_dim
        # C is 1 x p_dim
        for j in range(m_dim):
            a_idx = (i_off)*m + j_off + j
            for k in range(p_dim):
                b_idx = (j_off + j)*p + k_off + k
                c_idx = (k_off + k)*n + i_off
                add_rank_term([(a_idx, 1)], [(b_idx, 1)], [(c_idx, 1)])
