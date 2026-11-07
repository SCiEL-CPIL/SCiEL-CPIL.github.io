<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    rng = random.Random(12345)

    all_candidates = list(range(1, 257))
    # Prefer larger numbers to reduce divisibility collisions
    large_pool = list(range(256, 127, -1))
    mid_pool = list(range(128, 63, -1))
    small_pool = list(range(64, 0, -1))

    best_set = []
    best_size = 0
    target_size = 10

    # Phase 1: Greedy construction with multiple orderings
    for restart in range(300):
        if time.time() > deadline:
            break

        # Build candidate ordering: mix large numbers first with some randomness
        candidates = list(large_pool)
        rng.shuffle(candidates)
        # Sprinkle in some mid numbers
        mid_sample = list(mid_pool)
        rng.shuffle(mid_sample)
        candidates = candidates[:128] + mid_sample[:32]

        current_set, current_sums = _greedy_build(
            candidates, target_size, deadline, rng
        )

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)

        # Phase 2: Annealing polish on the greedy result
        polished_set, polished_sums = _anneal(
            current_set, current_sums, all_candidates, deadline, rng
        )
        if len(polished_set) > best_size:
            best_size = len(polished_set)
            best_set = list(polished_set)

        if best_size >= target_size:
            break

    # Final verification of best_set
    if best_set:
        verify_sums = _rebuild_sums(best_set)
        if not _check_antichain(verify_sums):
            # Fallback to a known small valid set
            best_set = [31, 32, 34]
            best_size = 3

    # Ensure all are Python ints and return a list
    return [int(x) for x in best_set]
=======
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    rng = random.Random(12345)

    all_candidates = list(range(1, 257))
    # Prefer larger numbers to reduce divisibility collisions
    large_pool = list(range(256, 127, -1))
    mid_pool = list(range(128, 63, -1))
    small_pool = list(range(64, 0, -1))

    best_set = []
    best_size = 0
    target_size = 10

    # Phase 1: Greedy construction with multiple orderings
    for restart in range(300):
        if time.time() > deadline:
            break

        # Build candidate ordering: mix large numbers first with some randomness
        candidates = list(large_pool)
        rng.shuffle(candidates)
        # Sprinkle in some mid numbers
        mid_sample = list(mid_pool)
        rng.shuffle(mid_sample)
        candidates = candidates[:128] + mid_sample[:32]

        current_set, current_sums = _greedy_build(
            candidates, target_size, deadline, rng
        )

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)

        # Phase 2: Annealing polish on the greedy result
        polished_set, polished_sums = _anneal(
            current_set, current_sums, all_candidates, deadline, rng
        )
        if len(polished_set) > best_size:
            best_size = len(polished_set)
            best_set = list(polished_set)

        if best_size >= target_size:
            break

    # Final verification of best_set
    if best_set:
        verify_sums = _rebuild_sums(best_set)
        if not _check_antichain(verify_sums):
            # Fallback to a known small valid set
            best_set = [31, 32, 34]
            best_size = 3

    # Ensure all are Python ints and return a list
    return [int(x) for x in best_set]
>>>>>>> REPLACE
