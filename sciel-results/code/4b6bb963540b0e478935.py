<<<<<<< SEARCH
def construct(n):
    """Return the shortest known superpermutation word for symbols 1..n."""
    start = time.time()
    deadline = start + _TIME_LIMIT - 5.0  # 5s safety margin

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1321231"
    if n == 4:
        return "14321243121431"
    if n == 5:
        return _classic_superperm(5)

    # For n >= 6, start with the classic construction
    word = _classic_superperm(n)

    # Attempt local improvement to shorten the word
    if time.time() < deadline:
        word = _try_local_improve(word, n, deadline)

    return word
=======
def construct(n):
    """Return the shortest known superpermutation word for symbols 1..n."""
    start = time.time()
    deadline = start + _TIME_LIMIT - 5.0  # 5s safety margin

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1321231"
    if n == 4:
        return "14321243121431"

    # For n=5, try B&B first (120 nodes is tractable), fall back to classic
    if n == 5:
        bb_result = _bfs_hamiltonian(5, deadline)
        if bb_result is not None:
            # Try to improve further
            if time.time() < deadline:
                bb_result = _try_local_improve(bb_result, 5, deadline)
            return bb_result
        else:
            return _classic_superperm(5)

    # For n >= 6, start with the classic construction
    word = _classic_superperm(n)

    # Attempt local improvement to shorten the word
    if time.time() < deadline:
        word = _try_local_improve(word, n, deadline)

    return word
>>>>>>> REPLACE
