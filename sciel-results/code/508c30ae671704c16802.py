    # Precompute neighbor sets
    all_neighbors = {}
    for i in range(total):
        all_neighbors[i] = set(_get_neighbors(i, q, n, radius))
