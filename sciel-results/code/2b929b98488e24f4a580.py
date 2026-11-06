<<<<<<< SEARCH
    best_score = N + 1
    best_mask = None

    # Initial structures to seed the search.
    seeds = [make_k666()]
    seeds.append(random_k4_free_graph())

    # Run several independent restarts; each gets a fair share of the budget.
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() >= deadline:
            break
        if r < len(seeds):
            adj = seeds[r][:]
        else:
            adj = random_k4_free_graph()
        if not is_k4_free(adj):
            adj = make_k666()
        cur = max_tri_free(adj)
        # Per-restart deadline: keep a little margin so we always return something.
        cur_deadline = min(deadline, time.time() + (deadline - start_time) * 0.0 + 15)
        local_best = anneal(adj, cur, deadline)
        if local_best < best_score:
            best_score = local_best
            best_mask = adj[:]

    if best_mask is None:
        best_mask = make_k666()
=======
    best_score = N + 1
    best_mask = None

    # Initial structures to seed the search.
    seeds = [make_k666(), make_c5_blowup(), make_petersen_blowup(), make_kneser_k4_free()]
    for _ in range(3):
        seeds.append(random_k4_free_graph())

    # Run several independent restarts; each gets a fair share of the budget.
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() >= deadline:
            break
        if r < len(seeds):
            adj = seeds[r][:]
        else:
            adj = random_k4_free_graph()
        if not is_k4_free(adj):
            adj = make_k666()
        cur = max_tri_free(adj)
        local_best, local_best_adj = anneal(adj, cur, deadline)
        if local_best < best_score:
            best_score = local_best
            best_mask = local_best_adj[:]

    if best_mask is None:
        best_mask = make_k666()
        best_score = max_tri_free(best_mask)

    # Final polish: intensive local search on the best candidate
    if time.time() < deadline - 2:
        polish_deadline = deadline - 1
        cur_score = best_score
        cur_adj = best_mask[:]
        # Multi-start local search: try each edge flip and keep improvements
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            for u in range(N):
                for v in range(u + 1, N):
                    if time.time() >= polish_deadline:
                        break
                    old_val = (cur_adj[u] >> v) & 1
                    flip(cur_adj, u, v)
                    if old_val == 0:
                        if adds_k4(cur_adj, u, v):
                            flip(cur_adj, u, v)
                            continue
                    new_score = max_tri_free(cur_adj)
                    if new_score < cur_score:
                        cur_score = new_score
                        improved = True
                    else:
                        flip(cur_adj, u, v)
                if time.time() >= polish_deadline:
                    break
        if cur_score < best_score:
            best_score = cur_score
            best_mask = cur_adj[:]
>>>>>>> REPLACE
