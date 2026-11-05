<<<<<<< SEARCH
def construct():
    """Return the best decomposition found within the time budget."""
    start = time.time()
    deadline = start + 115  # safety margin for 120s limit

    # Try known algorithms first
    result = _attempt_known_algorithms()
    if result is not None:
        return result

    # Try local search for ranks below schoolbook
    # Given the complexity, we'll skip extensive search and return schoolbook.
    # But let's at least try rank 62, 61 with a very short budget.
    for target_rank in [SCHOOLBOOK_RANK - 1, SCHOOLBOOK_RANK - 2]:
        if time.time() > deadline - 5:
            break
        budget = min(30, deadline - time.time() - 2)
        if budget <= 0:
            break
        result = _try_local_search(target_rank, budget, seed=target_rank)
        if result is not None:
            return result

    # Fall back to schoolbook
    return _schoolbook()
=======
def construct():
    """Return the best decomposition found within the time budget."""
    start = time.time()
    deadline = start + 115  # safety margin for 120s limit

    # Try block decomposition first (rank 21)
    result = _block_decomposition()
    if result is not None and _validate(result['U'], result['V'], result['W']):
        return result

    # Try known algorithms first
    result = _attempt_known_algorithms()
    if result is not None:
        return result

    # Try local search for ranks below schoolbook
    # Given the complexity, we'll skip extensive search and return schoolbook.
    # But let's at least try rank 62, 61 with a very short budget.
    for target_rank in [SCHOOLBOOK_RANK - 1, SCHOOLBOOK_RANK - 2]:
        if time.time() > deadline - 5:
            break
        budget = min(30, deadline - time.time() - 2)
        if budget <= 0:
            break
        result = _try_local_search(target_rank, budget, seed=target_rank)
        if result is not None:
            return result

    # Fall back to schoolbook
    return _schoolbook()
>>>>>>> REPLACE
