<<<<<<< SEARCH
    # At this point col should equal rank (50). Assert for safety.
    # (kept as light check; no print on success path)

    return {"U": U, "V": V, "W": W}
=======
    # At this point col should equal rank (50). Assert for safety.
    # (kept as light check; no print on success path)
    assert col == rank, f"Rank mismatch: col={col}, rank={rank}"

    _check_deadline()
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
