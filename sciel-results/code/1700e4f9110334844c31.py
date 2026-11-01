for j in range(2):
    for k_local in range(2):
        v_idx = j * p + (k0 + k_local)
        s_idx = j * 2 + k_local
        V[v_idx][col] = strassen_V[t][s_idx]
