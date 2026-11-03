    # 4x4x3 part: trivial
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                all_terms.extend(add_trivial_term(i, j, k))
