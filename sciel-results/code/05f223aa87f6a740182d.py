<<<<<<< SEARCH
    # Partition: [4, 1]
    blocks = [(0, 4), (4, 5)]
    
    for r1, r2 in blocks:
        for c1, c2 in blocks:
            for k1, k2 in blocks:
                if time.time() > deadline:
                    break
                ar = r2 - r1
                ac = c2 - c1
                bc = k2 - k1
                
                if ar == 4 and ac == 4 and bc == 4:
                    # Decompose 4x4x4 into four 2x2x2 Strassen blocks
                    for ri in range(0, 4, 2):
                        for ci in range(0, 4, 2):
                            for ki in range(0, 4, 2):
                                add_strassen_2x2x2(r1 + ri, c1 + ci, k1 + ki)
                else:
                    add_schoolbook_block(r1, r2, c1, c2, k1, k2)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
=======
    # Try multiple partitions and keep the best
    best_terms = None
    best_rank = 1000
    
    partitions_to_try = [
        [(0, 4), (4, 5)],
        [(0, 3), (3, 5)],
        [(0, 2), (2, 4), (4, 5)],
    ]
    
    for blocks in partitions_to_try:
        if time.time() > deadline - 5:
            break
        terms_trial = []
        
        def add_term_trial(U_dict, V_dict, W_dict):
            if time.time() > deadline:
                return
            U_dict = {k: v for k, v in U_dict.items() if v != 0}
            V_dict = {k: v for k, v in V_dict.items() if v != 0}
            W_dict = {k: v for k, v in W_dict.items() if v != 0}
            if U_dict and V_dict and W_dict:
                terms_trial.append((U_dict, V_dict, W_dict))
        
        for r1, r2 in blocks:
            for c1, c2 in blocks:
                for k1, k2 in blocks:
                    if time.time() > deadline:
                        break
                    ar = r2 - r1
                    ac = c2 - c1
                    bc = k2 - k1
                    
                    if ar >= 2 and ac >= 2 and bc >= 2:
                        # Decompose into 2x2x2 Strassen blocks where possible
                        for ri in range(0, ar, 2):
                            for ci in range(0, ac, 2):
                                for ki in range(0, bc, 2):
                                    if ri + 2 <= ar and ci + 2 <= ac and ki + 2 <= bc:
                                        add_strassen_2x2x2(r1 + ri, c1 + ci, k1 + ki)
                                    else:
                                        # Remaining parts: schoolbook
                                        for ii in range(ri + 2, ar):
                                            for jj in range(ci + 2, ac):
                                                for kk in range(ki + 2, bc):
                                                    u_idx = (r1 + ii) * m + (c1 + jj)
                                                    v_idx = (c1 + jj) * p + (k1 + kk)
                                                    w_idx = (k1 + kk) * n + (r1 + ii)
                                                    add_term_trial({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
                                        # Also handle edge cases where only one dimension has remainder
                                        if ri + 2 > ar and ci + 2 <= ac and ki + 2 <= bc:
                                            pass  # already handled
                                        elif ri + 2 <= ar and ci + 2 > ac and ki + 2 <= bc:
                                            pass
                                        elif ri + 2 <= ar and ci + 2 <= ac and ki + 2 > bc:
                                            pass
                    else:
                        add_schoolbook_block(r1, r2, c1, c2, k1, k2)
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        if len(terms_trial) < best_rank:
            best_rank = len(terms_trial)
            best_terms = terms_trial[:]
    
    if best_terms is not None:
        terms.clear()
        terms.extend(best_terms)
>>>>>>> REPLACE
