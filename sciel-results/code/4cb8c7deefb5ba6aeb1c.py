for c_block_idx in range(4):
    coeffs = recon_coeffs[c_block_idx]
    gamma = 0.0
    for (s_idx, g) in coeffs:
        if s_idx == s:
            gamma = g
            break
    if gamma != 0.0:
        br = c_block_idx // 2
        bc = c_block_idx % 2
        i_glob = br * bs + u
        j_glob = bc * bs + j
        w_row = j_glob * n + i_glob
        W[w_row][r_idx] += gamma
