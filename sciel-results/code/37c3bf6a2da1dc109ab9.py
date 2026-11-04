    # Part 2: A2 (2x1) * B2 (1x8)
    # This is an outer product, which has rank 1.
    # U has 1 at (i, 2) for i=0,1
    # V has 1 at (2, k) for all k
    # W has 1 at (k, i) for all k,i
    U[0 * m + 2][col] = 1
    U[1 * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    if not _elapsed_ok():
        return _schoolbook()
