    # Greedy initialization to get a good lower bound
    greedy_set = []
    selected_mask = 0
    # Sort by degree ascending for greedy
    order_asc = sorted(range(n), key=lambda x: degrees[x])
    for v in order_asc:
        if selected_mask & adj[v]:
            continue
        greedy_set.append(v)
        selected_mask |= (1 << v)
    
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = greedy_set[:]
