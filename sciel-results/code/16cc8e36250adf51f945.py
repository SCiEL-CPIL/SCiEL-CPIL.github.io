<<<<<<< SEARCH
    # Try to reduce rank (limited time)
    if time.time() < deadline - 5:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_time=deadline - time.time() - 5)

    # Final verification safety net
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _build_strassen_114()
=======
    # Try to reduce rank (limited time)
    if time.time() < deadline - 5:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_time=deadline - time.time() - 5)

    # Final verification safety net (with deadline guard)
    if time.time() < deadline - 1:
        if not _verify_decomposition(U, V, W, n, m, p):
            U, V, W = _build_strassen_114()
>>>>>>> REPLACE
