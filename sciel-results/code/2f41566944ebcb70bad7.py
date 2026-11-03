    ranks_to_try = list(range(21, 63))
    seeds_per_rank = 20
    # ...
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        for seed in range(seeds_per_rank):
            if time.time() > deadline - 5:
                break
            result = random_search(r, seed=1000 + r * 100 + seed,
                                   max_iters=100,
                                   deadline=deadline - 5)
            if result is not None:
                U_try, V_try, W_try = result
                if verify(U_try, V_try, W_try, tol=1e-7):
                    best_result = (U_try, V_try, W_try)
                    best_rank = r
                    break
        if best_rank < n * m * p:
            continue
