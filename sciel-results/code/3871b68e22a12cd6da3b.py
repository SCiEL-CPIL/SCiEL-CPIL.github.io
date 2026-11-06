    # ... inside construct() ...
    
    # Store top greedy paths
    top_greedy_paths = []
    
    for i in range(greedy_runs):
        # ...
        if len(path) > len(best_path):
            best_path = path
        # Maintain a list of top 10
        top_greedy_paths.append(path)
        top_greedy_paths.sort(key=lambda p: len(p), reverse=True)
        top_greedy_paths = top_greedy_paths[:10]

    # Beam Search
    def beam_search_with_initial(width, max_steps, initial_paths):
        beams = []
        for p in initial_paths:
            if not p: continue
            used = set(p)
            # Reconstruct forbidden
            forbidden = set()
            for i in range(len(p) - 1):
                forbidden.update(neighbors[p[i]])
            score = get_future_options(p[-1], used, forbidden)
            beams.append((score, p, used, forbidden))
        
        # Pad with random starts if needed
        while len(beams) < width:
            start_v = random.randint(0, 2**N - 1)
            path = [start_v]
            used = {start_v}
            forbidden = set()
            score = get_future_options(start_v, used, forbidden)
            beams.append((score, path, used, forbidden))
            
        # ... rest of beam search ...
