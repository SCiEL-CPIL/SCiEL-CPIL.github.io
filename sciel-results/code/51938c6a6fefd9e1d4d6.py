<<<<<<< SEARCH
def construct(n):
    if n == 1:
        return "1"
    if n <= 4:
        # Classic construction is optimal for n<=4
        word = _classic_superperm(n)
        return "".join(map(str, word))

    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    all_perms = set(itertools.permutations(range(1, n + 1)))

    # Generate candidate words from different heuristic modes within the
    # same constructive family, plus the classic construction as a baseline.
    candidates = []

    # Baseline: classic construction
    classic = _classic_superperm(n)
    candidates.append(classic)

    # Greedy constructive passes with different priority modes
    for mode in ('overlap', 'reuse', 'lex'):
        if time.time() > deadline - 10:
            break
        w = _greedy_construct(n, mode, deadline, start_time)
        if w:
            candidates.append(w)

    # Pick the shortest valid candidate
    best_word = None
    best_len = None
    for w in candidates:
        if not w:
            continue
        # validate
        if len(w) < n:
            continue
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i + n]))
        if not all_perms.issubset(seen):
            continue
        if best_len is None or len(w) < best_len:
            best_len = len(w)
            best_word = w

    if best_word is None:
        # fallback: classic is always valid
        best_word = classic
        best_len = len(classic)

    # Deletion polish on the best candidate if we have time
    if (time.time() - start_time) < deadline - 8 and best_len > n:
        polished = _deletion_polish(list(best_word), n, all_perms,
                                    deadline, start_time)
        # validate polished
        if len(polished) >= n:
            seen = set()
            for i in range(len(polished) - n + 1):
                seen.add(tuple(polished[i:i + n]))
            if all_perms.issubset(seen) and len(polished) < best_len:
                best_word = polished
                best_len = len(polished)

    return "".join(map(str, best_word))
=======
def _validate_superperm(word, n):
    """Check if word contains all n-permutations as consecutive windows."""
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        seen.add(tuple(word[i:i + n]))
    all_perms = set(itertools.permutations(range(1, n + 1)))
    return all_perms.issubset(seen)

def construct(n):
    if n == 1:
        return "1"
    if n <= 4:
        # Classic construction is optimal for n<=4
        word = _interleaved_superperm(n)
        if _validate_superperm(word, n):
            return "".join(map(str, word))
        word = _classic_superperm(n)
        return "".join(map(str, word))

    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    all_perms = set(itertools.permutations(range(1, n + 1)))

    # Generate candidate words from different constructive approaches
    candidates = []

    # Candidate 1: Interleaved construction (new)
    try:
        interleaved = _interleaved_superperm(n)
        if interleaved and _validate_superperm(interleaved, n):
            candidates.append(interleaved)
    except Exception:
        pass

    # Candidate 2: Classic construction (baseline)
    try:
        classic = _classic_superperm(n)
        if classic and _validate_superperm(classic, n):
            candidates.append(classic)
    except Exception:
        pass

    # Candidate 3+: Greedy constructive passes with different priority modes
    for mode in ('overlap', 'reuse', 'lex'):
        if time.time() > deadline - 15:
            break
        try:
            w = _greedy_construct(n, mode, deadline, start_time)
            if w and _validate_superperm(w, n):
                candidates.append(w)
        except Exception:
            pass

    # Pick the shortest valid candidate
    best_word = None
    best_len = None
    for w in candidates:
        if not w:
            continue
        if len(w) < n:
            continue
        if best_len is None or len(w) < best_len:
            best_len = len(w)
            best_word = w

    if best_word is None:
        # fallback: try classic again
        try:
            best_word = _classic_superperm(n)
            best_len = len(best_word)
        except Exception:
            # Last resort: brute force small case
            best_word = list(range(1, n + 1))
            best_len = n

    # Deletion polish on the best candidate if we have time
    if (time.time() - start_time) < deadline - 10 and best_len > n:
        try:
            polished = _deletion_polish(list(best_word), n, all_perms,
                                        deadline, start_time)
            if polished and len(polished) >= n and _validate_superperm(polished, n):
                if len(polished) < best_len:
                    best_word = polished
                    best_len = len(polished)
        except Exception:
            pass

    return "".join(map(str, best_word))
>>>>>>> REPLACE
