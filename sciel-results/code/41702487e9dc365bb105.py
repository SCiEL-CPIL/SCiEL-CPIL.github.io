<<<<<<< SEARCH
def construct():
    best = _baseline_network()
    best_len = len(best)
    if _verify(best):
        pass  # sanity; baseline is known valid

    pairs = _all_pairs()
    pair_index = {p: k for k, p in enumerate(pairs)}
    n_pairs = len(pairs)

    # Greedy deletion: repeatedly try to remove a comparator if still valid.
    cur = list(best)
    # Try each position; iterate to a fixpoint.
    changed = True
    while changed:
        changed = False
        for idx in range(len(cur) - 1, -1, -1):
            if _elapsed() > TIME_LIMIT:
                break
            cand = cur[:idx] + cur[idx + 1:]
            if _verify(cand):
                cur = cand
                changed = True
        if _elapsed() > TIME_LIMIT:
            break

    if len(cur) < best_len:
        best = cur
        best_len = len(cur)

    # Local swap search: replace one comparator with another to enable more deletions.
    # We run a limited number of random restarts / iterations within budget.
    import random
    rng = random.Random(12345)
    cur = list(best)

    def try_swap_remove(start):
        nonlocal cur, best, best_len
        n = len(start)
        for _ in range(400):
            if _elapsed() > TIME_LIMIT:
                return
            a = rng.randrange(n)
            b = rng.randrange(n)
            if a == b:
                continue
            cand = list(start)
            # swap positions a and b
            cand[a], cand[b] = cand[b], cand[a]
            # try to delete a random comparator
            d = rng.randrange(len(cand))
            cand2 = cand[:d] + cand[d + 1:]
            if _verify(cand2):
                cur = cand2
                best_len = len(cur)
                best = list(cur)
                # continue from here
                start = cur
                n = len(start)
            else:
                # try just the swap (same length) as a stepping stone occasionally
                if rng.random() < 0.3 and _verify(cand):
                    start = cand
                    n = len(start)

    try_swap_remove(cur)

    # Final: if we still have time, do a few more pure-deletion passes on best.
    cur = list(best)
    for _ in range(3):
        if _elapsed() > TIME_LIMIT:
            break
        changed = True
        while changed and _elapsed() < TIME_LIMIT:
            changed = False
            for idx in range(len(cur) - 1, -1, -1):
                cand = cur[:idx] + cur[idx + 1:]
                if _verify(cand):
                    cur = cand
                    changed = True
                    if len(cur) < best_len:
                        best_len = len(cur)
                        best = list(cur)
                    break

    # Ensure we return a valid network (fallback to baseline if something went wrong).
    result = best if len(best) <= 120 else _baseline_network()
    return result
=======
def construct():
    best = _baseline_network()
    best_len = len(best)
    
    # Verify baseline with fast method
    if _verify_fast(best):
        pass  # sanity; baseline is known valid
    else:
        # Fallback: use a known valid network
        # For now, assume baseline is valid
        pass

    # Greedy deletion: repeatedly try to remove a comparator if still valid.
    cur = list(best)
    # Try each position; iterate to a fixpoint.
    changed = True
    while changed:
        changed = False
        for idx in range(len(cur) - 1, -1, -1):
            if _elapsed() > TIME_LIMIT:
                break
            cand = cur[:idx] + cur[idx + 1:]
            if _verify_fast(cand):
                cur = cand
                changed = True
        if _elapsed() > TIME_LIMIT:
            break

    if len(cur) < best_len:
        best = cur
        best_len = len(cur)

    # Local search: try swapping and deleting to find better configurations
    import random
    rng = random.Random(12345)
    cur = list(best)
    
    # Try to improve by random mutations
    max_iter = 2000
    for _ in range(max_iter):
        if _elapsed() > TIME_LIMIT:
            break
        n = len(cur)
        if n == 0:
            break
        
        # Choose a random mutation type
        r = rng.random()
        cand = list(cur)
        
        if r < 0.5:
            # Try to delete a random comparator
            if len(cand) > 0:
                d = rng.randrange(len(cand))
                cand = cand[:d] + cand[d + 1:]
        elif r < 0.8:
            # Try to swap two comparators
            if len(cand) >= 2:
                a = rng.randrange(len(cand))
                b = rng.randrange(len(cand))
                if a != b:
                    cand[a], cand[b] = cand[b], cand[a]
        else:
            # Try to replace one comparator with a random other
            if len(cand) > 0:
                pos = rng.randrange(len(cand))
                i = rng.randrange(N)
                j = rng.randrange(i + 1, N)
                if (i, j) != cand[pos]:
                    cand[pos] = (i, j)
        
        # Verify the candidate
        if _verify_fast(cand):
            if len(cand) < len(cur):
                cur = cand
            elif len(cand) == len(cur) and rng.random() < 0.3:
                # Accept same-length improvements with some probability (for diversity)
                cur = cand
    
    if len(cur) < best_len:
        best = cur
        best_len = len(cur)

    # Final: if we still have time, do a few more pure-deletion passes on best.
    cur = list(best)
    for _ in range(5):
        if _elapsed() > TIME_LIMIT:
            break
        changed = True
        while changed and _elapsed() < TIME_LIMIT:
            changed = False
            for idx in range(len(cur) - 1, -1, -1):
                cand = cur[:idx] + cur[idx + 1:]
                if _verify_fast(cand):
                    cur = cand
                    changed = True
                    if len(cur) < best_len:
                        best_len = len(cur)
                        best = list(cur)
                    break

    # Ensure we return a valid network (fallback to baseline if something went wrong).
    result = best if len(best) <= 120 else _baseline_network()
    return result
>>>>>>> REPLACE
