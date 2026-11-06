    # Inside the loop for target in range...
    # Try a few random variations
    found_valid = False
    for attempt in range(5):
        adj_try = adj[:]
        n_try = n_base
        # Shuffle vertices for greedy selection?
        # Or just use different sorting criteria
        
        for k in range(target - n_base):
            new_v = n_try
            adj_try.append(0)
            n_try += 1
            # Randomized greedy
            vertices = list(range(n_try - 1))
            if attempt % 2 == 0:
                vertices.sort(key=lambda v: popcount(adj_try[v]), reverse=True)
            else:
                random.shuffle(vertices)
            # ... same logic
            
        saturate_edges(adj_try, n_try)
        # ... check alpha
        
        if alpha < 11:
            found_valid = True
            best_result = ...
            break
            
    if found_valid:
        break
