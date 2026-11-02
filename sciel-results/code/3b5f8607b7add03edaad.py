strassen_terms = _strassen_2x2x2_terms()
for term in strassen_terms:
    for (u_rel, v_rel, w_rel, coeff) in term:
        ...
        all_terms.append((u_idx, v_idx, w_idx, coeff))
