for local_k in range(2):
    for local_i in range(2):
        coeff = W2[local_k * 2 + local_i][c]
        ...
        cr, cc = C_blocks[c]
        global_k = 2 * cr + local_k
        global_i = 2 * cc + local_i
        W4[global_k * 4 + global_i][col] += coeff
