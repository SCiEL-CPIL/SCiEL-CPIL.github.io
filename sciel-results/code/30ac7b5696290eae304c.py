<<<<<<< SEARCH
    num_extra = _N - 11
    dists = generate_blowup_distributions(num_extra, 11)
    random.shuffle(dists)

    # also try a few explicit high-value distributions first
    extra_dists = []
    for d in dists:
        extra_dists.append(d)
    random.shuffle(extra_dists)

    for dist in extra_dists:
        if time.time() > deadline:
            break
        edges, cnt = _trial(dist, start_time)
        if edges is not None and cnt > best_count:
            best_count = cnt
            best_edges = edges
=======
    num_extra = _N - 11
    dists = generate_blowup_distributions(num_extra, 11)
    
    # Sort by "balance" - more balanced distributions tend to give denser graphs
    # A distribution where all parts have similar sizes is more balanced
    def balance_score(d):
        # Penalize variance in part sizes (including the +1 for original)
        sizes = [1 + x for x in d]
        mean = sum(sizes) / len(sizes)
        var = sum((s - mean) ** 2 for s in sizes)
        return var
    
    dists.sort(key=balance_score)
    
    # Interleave: try some balanced, some random
    extra_dists = list(dists)
    random.shuffle(extra_dists[len(extra_dists)//2:])
    
    for dist in extra_dists:
        if time.time() > deadline:
            break
        edges, cnt = _trial(dist, start_time)
        if edges is not None and cnt > best_count:
            best_count = cnt
            best_edges = edges
>>>>>>> REPLACE
