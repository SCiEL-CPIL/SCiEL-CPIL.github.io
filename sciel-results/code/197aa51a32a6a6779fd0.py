<<<<<<< SEARCH
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = random.Random(42)

    # Baseline
    U, V, W = baseline_66()
    r = 66

    # Local search to drop/merge redundant terms and HALS refine
    r, U, V, W = local_search(U, V, W, r, deadline, rng)

    assert verify(U, V, W, r), "Final decomposition is not exact!"

    return {"U": U, "V": V, "W": W}
=======
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = random.Random(42)

    # Baseline
    U, V, W = baseline_66()
    r = 66

    # Local search to drop redundant terms
    r, U, V, W = local_search(U, V, W, r, deadline, rng)

    # Final verification
    if not verify(U, V, W, r):
        # Fallback to baseline if search broke it
        U, V, W = baseline_66()
        r = 66
        if not verify(U, V, W, r):
            raise ValueError("Baseline is invalid")

    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
