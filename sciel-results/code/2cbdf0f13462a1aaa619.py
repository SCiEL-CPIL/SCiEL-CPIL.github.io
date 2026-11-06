<<<<<<< SEARCH
    # Apply local search to top candidates
    top_k = min(5, len(population))
    sorted_idx = np.argsort(scores)[:top_k]
    for idx in sorted_idx:
        improved_ind, improved_score = _local_search(population[idx], max_iters=200, rng=rng)
        if improved_score < best_score:
            best_score = improved_score
            best_individual = improved_ind
        population[idx] = improved_ind
        scores[idx] = improved_score
=======
    # Apply local search to top candidates
    top_k = min(10, len(population))
    sorted_idx = np.argsort(scores)[:top_k]
    for idx in sorted_idx:
        improved_ind, improved_score = _local_search(population[idx], max_iters=200, rng=rng)
        if improved_score < best_score:
            best_score = improved_score
            best_individual = improved_ind
        population[idx] = improved_ind
        scores[idx] = improved_score
>>>>>>> REPLACE

