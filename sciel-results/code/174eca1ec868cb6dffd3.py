<<<<<<< SEARCH
    # Phase 2: Multiple random-start greedy builds with local search / anneal
    while time.time() < deadline:
        adj = [0] * N
        order = all_edges[:]
        rng.shuffle(order)
        _greedy_from_order(adj, order)
        adj, c = _local_search(adj, max_iter=120, deadline=deadline, rng=rng)
        refresh_best(adj)
        if time.time() < deadline:
            adj, c = _anneal_from(adj, deadline, rng)
            refresh_best(adj)
=======
    # Phase 2: Multiple random-start greedy builds with local search / anneal
    while time.time() < deadline:
        adj = [0] * N
        order = all_edges[:]
        rng.shuffle(order)
        _greedy_from_order(adj, order)
        adj, c = _local_search(adj, max_iter=150, deadline=deadline, rng=rng)
        refresh_best(adj)
        if time.time() < deadline:
            # Give annealing a limited window
            anneal_deadline = min(deadline, time.time() + 15)
            adj, c = _anneal_from(adj, anneal_deadline, rng)
            refresh_best(adj)
>>>>>>> REPLACE
