<<<<<<< SEARCH
def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS

    # Start with the structured rank-44 decomposition
    U44, V44, W44 = _construct_structured_decomposition()
    best_U, best_V, best_W = U44.copy(), V44.copy(), W44.copy()
    best_rank = 44

    # Verify the initial decomposition
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fallback: schoolbook rank-48 decomposition
        U48 = np.zeros((n * m, 48))
        V48 = np.zeros((m * p, 48))
        W48 = np.zeros((p * n, 48))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U48[i * m + j, col] = 1
                    V48[j * p + k, col] = 1
                    W48[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U48, V48, W48
        best_rank = 48

    # Try to reduce rank
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 20:
            break
        U_try, V_try, W_try = _try_reduce_rank(
            best_U, best_V, best_W, n, m, p,
            max_reductions=best_rank - target_rank,
            time_budget=15
        )
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
=======
def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS

    # Try the rank-36 decomposition first
    U36, V36, W36 = _construct_rank_36_attempt()
    if _verify_decomposition(U36, V36, W36, n, m, p, tol=1e-9):
        best_U, best_V, best_W = U36.copy(), V36.copy(), W36.copy()
        best_rank = 36
    else:
        # Fall back to rank-44
        U44, V44, W44 = _construct_structured_decomposition()
        best_U, best_V, best_W = U44.copy(), V44.copy(), W44.copy()
        best_rank = 44

    # Verify the initial decomposition
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-9):
        # Fallback: schoolbook rank-48 decomposition
        U48 = np.zeros((n * m, 48))
        V48 = np.zeros((m * p, 48))
        W48 = np.zeros((p * n, 48))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U48[i * m + j, col] = 1
                    V48[j * p + k, col] = 1
                    W48[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U48, V48, W48
        best_rank = 48

    # Try to reduce rank using single-term elimination
    outers = _compute_outer_products(best_U, best_V, best_W, n, m, p)
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        U_try, V_try, W_try = _try_eliminate_one_term(best_U, best_V, best_W, n, m, p, outers)
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]
            outers = _compute_outer_products(best_U, best_V, best_W, n, m, p)
        else:
            break

    # Try to reduce rank using multi-term elimination
    if time.time() < deadline - 20:
        outers = _compute_outer_products(best_U, best_V, best_W, n, m, p)
        U_try, V_try, W_try = _try_eliminate_multiple_terms(best_U, best_V, best_W, n, m, p, outers, max_k=3)
        if U_try.shape[1] < best_rank:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = U_try.shape[1]

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
>>>>>>> REPLACE
