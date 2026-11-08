<<<<<<< SEARCH
    # ---- Phase 4: local search from best ----
    if time.time() < deadline - 5:
        opt_U, opt_score = local_optimize(best_U, deadline - 5, rng, max_iters=600)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    # ---- Phase 5: local search from diverse candidates ----
    seen = set()
    for cand in diverse:
        if time.time() > deadline - 3:
            break
        key = tuple(cand)
        if key in seen:
            continue
        seen.add(key)
        if key == tuple(best_U):
            continue
        opt_U, opt_score = local_optimize(cand, deadline - 3, rng, max_iters=250)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    # ---- Phase 6: final polish ----
    if time.time() < deadline - 1:
        opt_U, opt_score = local_optimize(best_U, deadline - 1, rng, max_iters=400)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    return best_U
=======
    # ---- Phase 4: greedy frequency-gap construction from best ----
    if time.time() < deadline - 15:
        greedy_U, greedy_score = _greedy_frequency_gap(best_U, deadline - 15, rng, max_size=18)
        if greedy_score > best_score:
            best_score = greedy_score
            best_U = greedy_U

    # ---- Phase 5: local search from best ----
    if time.time() < deadline - 8:
        opt_U, opt_score = local_optimize(best_U, deadline - 8, rng, max_iters=800)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    # ---- Phase 6: local search from diverse candidates ----
    seen = set()
    for cand in diverse:
        if time.time() > deadline - 5:
            break
        key = tuple(cand)
        if key in seen:
            continue
        seen.add(key)
        if key == tuple(best_U):
            continue
        opt_U, opt_score = local_optimize(cand, deadline - 5, rng, max_iters=300)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    # ---- Phase 7: final polish ----
    if time.time() < deadline - 1:
        opt_U, opt_score = local_optimize(best_U, deadline - 1, rng, max_iters=500)
        if opt_score > best_score:
            best_score = opt_score
            best_U = opt_U

    return best_U
>>>>>>> REPLACE
