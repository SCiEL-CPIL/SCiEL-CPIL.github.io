<<<<<<< SEARCH
    # Attempt rank 10
    if time.time() < deadline:
        result10 = _try_rank10(deadline)
        if result10 is not None:
            U, V, W = result10
            best = {"U": U, "V": V, "W": W}

    return best
=======
    # Attempt rank 10
    if time.time() < deadline:
        result10 = _try_rank10(deadline)
        if result10 is not None:
            U, V, W = result10
            # Final verification with exact check
            if _verify_exact(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}

    return best
>>>>>>> REPLACE
