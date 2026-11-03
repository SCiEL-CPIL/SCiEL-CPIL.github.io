def merge_by_u_rank1(terms_in):
    groups = defaultdict(list)
    for u, v, w in terms_in:
        key = tuple(sorted(u.items()))
        groups[key].append((v, w))
    
    merged = []
    for key, vw_list in groups.items():
        u = dict(key)
        if len(vw_list) == 1:
            v, w = vw_list[0]
            merged.append((u, v, w))
        else:
            # Sum v w^T
            M = np.zeros((len(vw_list[0][0]), len(vw_list[0][1])))
            # Note: v and w are dicts. We need to convert to vectors.
            # This is inefficient.
            # We should keep v and w as vectors in the terms?
            # The current terms are dicts.
            # We can convert to vectors for the check.
            
            # Determine dimensions
            # m = max index in v + 1
            # p = max index in w + 1
            m_max = 0
            p_max = 0
            for v, w in vw_list:
                for idx in v:
                    m_max = max(m_max, idx)
                for idx in w:
                    p_max = max(p_max, idx)
                    
            m_dim = m_max + 1
            p_dim = p_max + 1
            
            M = np.zeros((m_dim, p_dim))
            for v, w in vw_list:
                v_vec = np.zeros(m_dim)
                w_vec = np.zeros(p_dim)
                for idx, c in v.items():
                    v_vec[idx] = c
                for idx, c in w.items():
                    w_vec[idx] = c
                M += np.outer(v_vec, w_vec)
                
            result = _check_rank_1_and_decompose(M)
            if result:
                v_new, w_new = result
                # Convert back to dict
                v_dict = {i: c for i, c in enumerate(v_new) if c != 0}
                w_dict = {i: c for i, c in enumerate(w_new) if c != 0}
                merged.append((u, v_dict, w_dict))
            else:
                # Keep original terms
                for v, w in vw_list:
                    merged.append((u, v, w))
                    
    return merged
