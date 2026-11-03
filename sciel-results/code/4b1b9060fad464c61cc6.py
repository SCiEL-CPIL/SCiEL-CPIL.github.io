    # Naive part for bottom row (i=2)
    # A[2, j] * B[j, k] -> C[2, k]
    # Using rank 1 decomposition for M_1,4,4
    for j in range(m):
        U[2 * m + j][col] += 1
    for j in range(m):
        for k in range(p):
            V[j * p + k][col] += 1
    for k in range(p):
        W[k * n + 2][col] += 1
    col += 1
