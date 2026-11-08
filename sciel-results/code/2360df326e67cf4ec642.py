def get_perm_matrix(n_block, n_local):
    # Returns P such that P[flat_rowmajor, kron_index] = 1
    # flat = I * (n_local * n_block) + J * n_local + r * (n_block / n_local?) 
    # Let's stick to 2x2 blocks of 2x2.
    # n_block = 2, n_local = 2.
    # Flat: i * 4 + j. i = 2I + r, j = 2J + c.
    # Flat = (2I+r)*4 + (2J+c) = 8I + 4r + 2J + c.
    # Kron: I * 4 + J * 2 + r * 1 + c? 
    # Kron vector of A (2x2 blocks of 2x2) is [vec(A00), vec(A01), vec(A10), vec(A11)]
    # vec(Aij) is 4 elements.
    # Index K = I * 4 + J * 4 + idx_local.
    # idx_local = r * 2 + c.
    # K = 4I + 4J + 2r + c.
    P = np.zeros((16, 16))
    for I in range(2):
        for J in range(2):
            for r in range(2):
                for c in range(2):
                    i = 2*I + r
                    j = 2*J + c
                    flat = i*4 + j
                    kron = I*4 + J*4 + r*2 + c
                    P[flat, kron] = 1
    return P
