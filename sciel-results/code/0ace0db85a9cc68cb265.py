<<<<<<< SEARCH
def construct():
    """
    Constructs a minimal-rank exact tensor decomposition for 4x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer entries.
    """
    start_time = time.time()
    deadline = start_time + 100

    n, m, p = DIMENSIONS

    # Try ranks from 32 to 48
    for r in [32, 34, 36, 38, 40, 42, 44, 46, 48]:
        if time.time() > deadline:
            break
        budget = min(15, deadline - time.time())
        if budget < 5:
            break
        res = _optimize_rank(r, time_budget=budget, seed=42 + r)
        if res is not None:
            if _verify_decomposition(res["U"], res["V"], res["W"], n, m, p):
                return res

    # Fallback: rank-49 Kronecker-Strassen
    if time.time() < deadline:
        return _strassen_kron_4x4(n, m, p)

    # Final fallback: schoolbook
    return _schoolbook(n, m, p)
=======
def construct():
    """
    Constructs a minimal-rank exact tensor decomposition for 4x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Try rank 48 first (closest to known 49), with multiple seeds
    for r in [48, 47, 46, 45, 44, 42, 40]:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 10:
            break
        # Multiple attempts with different seeds
        for attempt in range(3):
            if time.time() > deadline:
                break
            budget = min(remaining - 5, 25)
            if budget < 5:
                break
            seed = 42 + r * 10 + attempt
            res = _optimize_rank(r, time_budget=budget, seed=seed, use_strassen_init=True)
            if res is not None:
                if _verify_decomposition(res["U"], res["V"], res["W"], n, m, p):
                    return res

    # Fallback: rank-49 Kronecker-Strassen
    if time.time() < deadline:
        return _strassen_kron_4x4(n, m, p)

    # Final fallback: schoolbook
    return _schoolbook(n, m, p)
>>>>>>> REPLACE
