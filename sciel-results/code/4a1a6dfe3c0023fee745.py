<<<<<<< SEARCH
    # Process each seed
    num_seeds = len(seeds)
    time_per_seed = (deadline - start) / num_seeds
    
    for idx, (name, (adj, deg)) in enumerate(seeds):
        seed_deadline = start + (idx + 1) * time_per_seed
        if time.time() > seed_deadline - 2:
            break
        
        # Greedy add phase
        adj, deg = _greedy_add(adj, deg, seed_deadline - 3)
        count = _edge_count(adj)
        
        # Local search phase
        rng = random.Random(42 + idx * 1000)
        adj, deg, count = _local_search(adj, deg, seed_deadline - 1, rng, max_iter=100000)
        
        # Final greedy add
        adj, deg = _greedy_add(adj, deg, seed_deadline - 0.5)
        count = _edge_count(adj)
        
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
=======
    # Process each seed
    num_seeds = len(seeds)
    time_per_seed = (deadline - start) / num_seeds
    
    for idx, (name, (adj, deg)) in enumerate(seeds):
        seed_deadline = start + (idx + 1) * time_per_seed
        if time.time() > seed_deadline - 2:
            break
        
        # Greedy add phase
        adj, deg = _greedy_add(adj, deg, seed_deadline - 3)
        count = _edge_count(adj)
        
        # Local search phase
        rng = random.Random(42 + idx * 1000)
        adj, deg, count = _local_search(adj, deg, seed_deadline - 1, rng, max_iter=150000)
        
        # Final greedy add
        adj, deg = _greedy_add(adj, deg, seed_deadline - 0.5)
        count = _edge_count(adj)
        
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
>>>>>>> REPLACE
