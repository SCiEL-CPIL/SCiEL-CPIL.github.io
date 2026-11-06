<<<<<<< SEARCH
    # Main search loop
    for it in range(50000):
        if time.time() > deadline - 3:
            break

        strat = it % 6

        if strat == 0:
            # Random single seeds
            A0 = [rng.randint(1, n)]
            B0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(A0, B0, "greedy")
        elif strat == 1:
            # A first
            A0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(A0, None, "A_first")
        elif strat == 2:
            # B first
            B0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(None, B0, "B_first")
        elif strat == 3:
            # Small random starts
            sa = rng.randint(1, 3)
            sb = rng.randint(1, 3)
            A0 = sorted(rng.sample(range(1, n + 1), sa))
            B0 = sorted(rng.sample(range(1, n + 1), sb))
            A, B, score = _attempt_build(A0, B0, "greedy")
        elif strat == 4:
            # Wider spacing seeds
            A0 = [rng.randint(1, 128)]
            B0 = [rng.randint(129, n)]
            A, B, score = _attempt_build(A0, B0, "greedy")
        else:
            # Number-theoretic seeds
            A0, B0 = _number_theoretic_seed(rng, n)
            if A0 is None:
                A0 = [rng.randint(1, n)]
                B0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(A0, B0, "greedy")

        if A is not None and score > best_score:
            best_score = score
            best_A = list(A)
            best_B = list(B)

            # Polish with exhaustive local search
            if time.time() < deadline - 15:
                best_A, best_B = _deep_polish(best_A, best_B, time_limit=8.0)
                best_score = len(best_A) + len(best_B)

    # Final validation
    if not _valid(best_A, best_B):
        best_A = [1]
        best_B = [2]
        best_score = 2

    return [best_A, best_B]
=======
    # Main search loop
    for it in range(50000):
        if time.time() > deadline - 10:
            break

        strat = it % 6

        if strat == 0:
            # Random single seeds
            A0 = [rng.randint(1, n)]
            B0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(A0, B0, "greedy")
        elif strat == 1:
            # A first
            A0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(A0, None, "A_first")
        elif strat == 2:
            # B first
            B0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(None, B0, "B_first")
        elif strat == 3:
            # Small random starts
            sa = rng.randint(1, 3)
            sb = rng.randint(1, 3)
            A0 = sorted(rng.sample(range(1, n + 1), sa))
            B0 = sorted(rng.sample(range(1, n + 1), sb))
            A, B, score = _attempt_build(A0, B0, "greedy")
        elif strat == 4:
            # Wider spacing seeds
            A0 = [rng.randint(1, 128)]
            B0 = [rng.randint(129, n)]
            A, B, score = _attempt_build(A0, B0, "greedy")
        else:
            # Number-theoretic seeds
            A0, B0 = _number_theoretic_seed(rng, n)
            if A0 is None:
                A0 = [rng.randint(1, n)]
                B0 = [rng.randint(1, n)]
            A, B, score = _attempt_build(A0, B0, "greedy")

        if A is not None and score > best_score:
            best_score = score
            best_A = list(A)
            best_B = list(B)

            # Polish with exhaustive local search
            if time.time() < deadline - 12:
                best_A, best_B = _deep_polish(best_A, best_B, time_limit=6.0)
                best_score = len(best_A) + len(best_B)

    # Final exhaustive polish using remaining time
    remaining = deadline - time.time()
    if remaining > 2:
        best_A, best_B = _deep_polish(best_A, best_B, time_limit=max(1.0, remaining - 2))
        best_score = len(best_A) + len(best_B)

    # Final validation
    if not _valid(best_A, best_B):
        best_A = [1]
        best_B = [2]
        best_score = 2

    return [best_A, best_B]
>>>>>>> REPLACE

