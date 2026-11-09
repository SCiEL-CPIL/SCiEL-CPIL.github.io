<<<<<<< SEARCH
    # Phase 2: Iterated Local Search
    while time.time() < deadline:
        k = rng.randint(2, 5)
        perturbed = _perturb(best_adj, rng, k)
        perturbed = _greedy_fill(perturbed, all_edges, rng)
        perturbed, c = _local_search(perturbed, max_iter=100, deadline=deadline, rng=rng)
        refresh_best(perturbed)
=======
    # Phase 2: Iterated Local Search
    while time.time() < deadline:
        k = rng.randint(2, 5)
        perturbed = _perturb(best_adj, rng, k)
        perturbed = _greedy_fill(perturbed, all_edges, rng)
        perturbed, c = _local_search(perturbed, max_iter=100, deadline=deadline, rng=rng)
        refresh_best(perturbed)
        
        # Phase 2b: Annealing polish on the current best
        if time.time() < deadline:
            polished, pc = _anneal_polish(best_adj, deadline, rng)
            refresh_best(polished)
>>>>>>> REPLACE
