<<<<<<< SEARCH
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0

    # Build the 84-rank decomposition with partition (1,2), (2,2,1), (2,2,2)
    U, V, W, rank = build_block_decomposition([1, 2], [2, 2, 1], [2, 2, 2])

    if verify_decomposition(U, V, W, n, m, p):
        # Try to reduce rank
        result = try_rank_reduction(U, V, W, n, m, p, rank, deadline)
        if result is not None:
            new_U, new_V, new_W, new_rank = result
            if new_rank < rank:
                return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1

    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
=======
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0

    # Search for the best block partition
    best_partition, best_rank = _find_best_block_partition(n, m, p, deadline - 10)

    if time.time() >= deadline - 5:
        # Not enough time, use the known good partition
        best_partition = ([1, 2], [2, 2, 1], [2, 2, 2])
        best_rank = 84

    n_sizes, m_sizes, p_sizes = best_partition
    U, V, W, rank = build_block_decomposition(n_sizes, m_sizes, p_sizes)

    if verify_decomposition(U, V, W, n, m, p):
        # Try to reduce rank
        result = try_rank_reduction(U, V, W, n, m, p, rank, deadline)
        if result is not None:
            new_U, new_V, new_W, new_rank = result
            if new_rank < rank:
                return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb), dtype=np.float64)
    V_fb = np.zeros((m * p, rank_fb), dtype=np.float64)
    W_fb = np.zeros((p * n, rank_fb), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1

    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
>>>>>>> REPLACE
