    def add_rank1(u_local, v_local, w_local, i_off, j_off, k_off, scale=1):
        nonlocal col
        for (i, j), val in u_local.items():
            if val != 0:
                u_idx = (i_off + i) * m + (j_off + j)
                U[u_idx][col] += val * scale
        for (j, k), val in v_local.items():
            if val != 0:
                v_idx = (j_off + j) * p + (k_off + k)
                V[v_idx][col] += val * scale
        for (k, i), val in w_local.items():
            if val != 0:
                w_idx = (k_off + k) * n + (i_off + i)
                W[w_idx][col] += val * scale
        col += 1
