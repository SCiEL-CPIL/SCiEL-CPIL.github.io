<<<<<<< SEARCH
    # Main loop: multiple restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    restart_deadline = time_limit * 0.50  # Leave more time for final local search

    seed = 42
    while time.time() - start_time < restart_deadline:
        random.seed(seed)
        np.random.seed(seed)
        greedy_blocks = greedy(seed)
        # Local search: remove then replace
        refined = local_search_remove(greedy_blocks)
        if time.time() - start_time < time_limit - 10:
            remaining = time_limit - (time.time() - start_time) - 10
            refined = local_search_replace(refined, min(2.0, max(0.1, remaining * 0.08)))
        if time.time() - start_time < time_limit - 8:
            refined = local_search_remove(refined)

        count = len(refined)
        if count < best_count:
            best_count = count
            best_blocks = refined

        seed += 1
        if best_count <= 70:  # Good enough, stop early
            break

    # Final polish on the best solution
    if best_blocks is not None and time.time() - start_time < time_limit - 5:
        polished = local_search_remove(best_blocks)
        if time.time() - start_time < time_limit - 4:
            remaining = time_limit - (time.time() - start_time) - 4
            polished = local_search_replace(polished, min(6.0, max(0.5, remaining * 0.5)))
        if time.time() - start_time < time_limit - 3:
            polished = local_search_remove(polished)
        if time.time() - start_time < time_limit - 2:
            remaining = time.time() - start_time
            # Use kick polish in the remaining time, but be careful not to exceed limit
            kick_budget = min(4.0, max(0.3, time_limit - 1 - remaining))
            if kick_budget > 0.1:
                polished = kick_polish(polished, kick_budget)
        if time.time() - start_time < time_limit - 1:
            polished = local_search_remove(polished)
        best_blocks = polished
=======
    # Main loop: multiple restarts with different seeds
    best_blocks = None
    best_count = float('inf')
    restart_deadline = time_limit * 0.45  # Leave more time for final local search

    seed = 42
    while time.time() - start_time < restart_deadline:
        random.seed(seed)
        np.random.seed(seed)
        greedy_blocks = greedy(seed)
        # Local search: remove then replace then single
        refined = local_search_remove(greedy_blocks)
        if time.time() - start_time < time_limit - 15:
            remaining = time_limit - (time.time() - start_time) - 15
            refined = local_search_replace(refined, min(1.5, max(0.1, remaining * 0.05)))
        if time.time() - start_time < time_limit - 12:
            refined = local_search_single(refined, min(1.0, max(0.1, (time_limit - (time.time() - start_time) - 12) * 0.1)))
        if time.time() - start_time < time_limit - 10:
            refined = local_search_remove(refined)

        count = len(refined)
        if count < best_count:
            best_count = count
            best_blocks = list(refined)

        seed += 1
        if best_count <= 70:  # Good enough, stop early
            break

    # Final polish on the best solution
    if best_blocks is not None and time.time() - start_time < time_limit - 5:
        polished = local_search_remove(best_blocks)
        if time.time() - start_time < time_limit - 4:
            remaining = time_limit - (time.time() - start_time) - 4
            polished = local_search_replace(polished, min(4.0, max(0.5, remaining * 0.3)))
        if time.time() - start_time < time_limit - 3:
            remaining = time_limit - (time.time() - start_time) - 3
            polished = local_search_single(polished, min(3.0, max(0.3, remaining * 0.3)))
        if time.time() - start_time < time_limit - 2:
            polished = local_search_remove(polished)
        if time.time() - start_time < time_limit - 1.5:
            remaining = time.time() - start_time
            kick_budget = min(3.0, max(0.2, time_limit - 1.5 - remaining))
            if kick_budget > 0.1:
                polished = kick_polish(polished, kick_budget)
        if time.time() - start_time < time_limit - 1:
            polished = local_search_remove(polished)
        best_blocks = polished
>>>>>>> REPLACE

