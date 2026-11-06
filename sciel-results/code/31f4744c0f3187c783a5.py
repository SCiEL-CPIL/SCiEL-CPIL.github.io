<<<<<<< SEARCH
        shuffled_remove = list(edge_set)
        rng.shuffle(shuffled_remove)
        for e_remove in shuffled_remove:
            if time.time() >= deadline:
                break
            shuffled_add = list(non_edges)
            rng.shuffle(shuffled_add)
            for e_add in shuffled_add:
                if e_add in edge_set:
                    continue
                new_score, improved_try = _try_swap(n, mask, deg, edge_set, e_remove, e_add, best_score)
                if improved_try:
                    best_score = new_score
                    best_edge_set = set(edge_set)
                    best_edge_set.discard(e_remove)
                    best_edge_set.add(tuple(sorted(e_add)))
                    current_score = new_score
                    improved = True
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(e_add)))
                    non_edges = _get_non_edges(n, edge_set)
                    break
            if improved:
                break
=======
        shuffled_remove = list(edge_set)
        rng.shuffle(shuffled_remove)
        for e_remove in shuffled_remove:
            if time.time() >= deadline:
                break
            shuffled_add = list(non_edges)
            rng.shuffle(shuffled_add)
            for e_add in shuffled_add:
                if e_add in edge_set:
                    continue
                new_score, improved_try = _try_swap(n, mask, deg, edge_set, e_remove, e_add, best_score, current_score)
                if improved_try:
                    best_score = new_score
                    best_edge_set = set(edge_set)
                    best_edge_set.discard(e_remove)
                    best_edge_set.add(tuple(sorted(e_add)))
                    current_score = new_score
                    improved = True
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(e_add)))
                    non_edges = _get_non_edges(n, edge_set)
                    break
            if improved:
                break
>>>>>>> REPLACE

