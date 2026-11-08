u[(i_off+1) * m + j_off] = a_scale  # a21
u[(i_off+1) * m + (j_off+1)] = a_scale  # a22
v[j_off * p + k_off] = b_scale  # b11
w[k_off * n + (i_off+1)] = c_scale  # c21
w[(k_off+1) * n + (i_off+1)] = -c_scale  # -c22
