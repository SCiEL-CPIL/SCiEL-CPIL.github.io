    strassen_steps = get_strassen_2x2_terms()
    
    for step in strassen_steps:
        A_local, B_local = step
        # ...
        for a_idx, a_coeff in A_local:
