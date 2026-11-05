<<<<<<< SEARCH
    # ---------- Main ----------
    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    # Try structured PGL for small q
    q_max_struct = min(best_n[0] - 2, 80)
    for curr_q in range(max(3, n_min - 1), q_max_struct + 1):
        if time.time() > deadline:
            break
        res = try_structured_pgl(curr_q, k, g)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            return best_result

    # Random PGL search
    q_cursor = max(3, n_min - 1)
    while q_cursor + 1 < best_n[0] and time.time() < deadline:
        budget = min(deadline - 2.0, time.time() + 1.5)
        res = random_search_pgl(q_cursor, k, g, budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
        q_cursor += 1

    return best_result
=======
    # ---------- Main ----------
    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    # Try structured PGL for small q
    q_max_struct = min(best_n[0] - 2, 100)
    for curr_q in range(max(3, n_min - 1), q_max_struct + 1):
        if time.time() > deadline - 30:
            break
        res = try_structured_pgl(curr_q, k, g)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            return best_result

    # Random PGL search - explore more q values with better budget
    q_cursor = max(3, n_min - 1)
    max_q = best_n[0] - 1
    time_per_q = 0.8  # seconds per q value
    while q_cursor < max_q and time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 0.5:
            break
        budget = min(remaining_time, time.time() + time_per_q)
        res = random_search_pgl(q_cursor, k, g, budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
        q_cursor += 1

    return best_result
>>>>>>> REPLACE
