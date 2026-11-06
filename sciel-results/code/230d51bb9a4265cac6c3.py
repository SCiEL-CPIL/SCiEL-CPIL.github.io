<<<<<<< SEARCH
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break

        # More restarts for larger n; the random graph is unlikely to be valid,
        # so local search must do the heavy lifting.
        max_attempts = 12 if n >= 19 else 20

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Vary edge probability around 0.5
            p = 0.35 + 0.3 * rng.random()

            adj_masks, edges = random_graph(n, p, rng)

            viol = find_violation(adj_masks, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_valid = True
                break  # found a valid graph for this n, move to next n

            # Local search to repair
            new_n, new_edges, new_valid = local_search(adj_masks, n, rng, deadline)
            if new_valid and new_n > best_n:
                best_n = new_n
                best_edges = new_edges
                best_valid = True
                break

        if best_valid and best_n >= n:
            # Already have a valid graph at this size or larger
            continue
=======
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break

        # More restarts for larger n; the random graph is unlikely to be valid,
        # so local search must do the heavy lifting.
        max_attempts = 20 if n >= 19 else 30

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Vary edge probability around 0.5
            p = 0.4 + 0.2 * rng.random()

            adj_masks, edges = random_graph(n, p, rng)

            viol = find_violation(adj_masks, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_valid = True
                break  # found a valid graph for this n, move to next n

            # Local search to repair
            new_n, new_edges, new_valid = local_search(adj_masks, n, rng, deadline)
            if new_valid and new_n > best_n:
                best_n = new_n
                best_edges = new_edges
                best_valid = True
                break

        if best_valid and best_n >= n:
            # Already have a valid graph at this size or larger
            continue
>>>>>>> REPLACE
