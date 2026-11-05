def add_schoolbook_col(U, V, W, k, rank_start):
    # k is fixed, j varies 0..3
    for j in range(4):
        # i=0
        r = rank_start + j*2
        U[0*4 + j][r] += 1
        V[j*5 + k][r] += 1
        W[k*2 + 0][r] += 1
        # i=1
        r = rank_start + j*2 + 1
        U[1*4 + j][r] += 1
        V[j*5 + k][r] += 1
        W[k*2 + 1][r] += 1
