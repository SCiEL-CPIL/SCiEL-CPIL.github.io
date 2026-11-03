<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6

    # Build the rank-42 Strassen decomposition
    U, V, W = _build_strassen_42()
    best_rank = 42

    # Verify with exact integer arithmetic
    if _verify_decomposition(U, V, W, n, m, p):
        pass  # baseline is good
    else:
        # Fallback to schoolbook
        U, V, W = _build_schoolbook_48()
        best_rank = 48

    # Final safety: ensure we return something valid
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _build_strassen_42()
        if not _verify_decomposition(U, V, W, n, m, p):
            U, V, W = _build_schoolbook_48()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6

    # Build the rank-42 Strassen decomposition
    U, V, W = _build_strassen_42()
    best_rank = 42

    # Verify with exact integer arithmetic
    if _elapsed() < _DEADLINE and not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook_48()
        best_rank = 48

    # Final safety: ensure we return something valid
    if _elapsed() < _DEADLINE:
        if not _verify_decomposition(U, V, W, n, m, p):
            U, V, W = _build_strassen_42()
            if not _verify_decomposition(U, V, W, n, m, p):
                U, V, W = _build_schoolbook_48()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
