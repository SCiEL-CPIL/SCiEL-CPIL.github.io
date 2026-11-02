def construct():
    start = time.time()
    deadline = start + 130.0
    n, m, p = DIMENSIONS

    # Schoolbook as fallback
    U0, V0, W0 = _schoolbook(n, m, p)
    best = (U0, V0, W0)
    best_rank = n * m * p

    # Try corrected block decomposition
    if time.time() < deadline:
        U, V, W = _kronecker_decompose_3x5x6()
        if _check_exact(U, V, W, n, m, p):
            best = (U, V, W)
            best_rank = U.shape[1]

    U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
