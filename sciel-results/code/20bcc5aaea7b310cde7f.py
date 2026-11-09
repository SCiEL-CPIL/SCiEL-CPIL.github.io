    # Add dynamic greedy result to initial candidates
    if time.time() < deadline:
        cand = greedy_mis_dynamic()
        cand = local_search(cand, max_iters=300)
        if len(cand) > len(best):
            best = cand
