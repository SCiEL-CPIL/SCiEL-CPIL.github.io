for i in i_group:
    for j in j_group:
        for k in k_group:
            add_term([(i * m + j, 1)], [(j * p + k, 1)], [(k * n + i, 1)])
