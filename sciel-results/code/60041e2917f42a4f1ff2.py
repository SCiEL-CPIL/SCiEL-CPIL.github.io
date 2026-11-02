    candidate_ranks = [110, 105, 100, 95, 90, 85, 80, 75, 70]
    for rank in candidate_ranks:
        if rank >= best_rank:
            continue
        if time.time() > deadline - 15:
            break
        res = _try_ilp_decomposition(n, m, p, rank, time_limit=12.0)
        if res is not None:
            best = res
            best_rank = rank
            break
