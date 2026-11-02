for local_k in range(2):
    k_global = 2 * blk + local_k
    for i in range(2):
        W[k_global * n + i][col] = W_block[local_k * 2 + i][s]
