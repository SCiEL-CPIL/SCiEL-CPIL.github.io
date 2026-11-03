def add_1x2_2x8(i_off, j_off, k_off):
    for j in range(2):
        for k in range(8):
            u = [(i_off*4 + j_off + j, 1)]
            v = [(j_off + j)*8 + k_off + k, 1]
            w = [(k_off + k)*3 + i_off, 1]
            add_term(u, v, w)
