<<<<<<< SEARCH
    # Main search loop
    best_count = 0
    best_adj = None
    
    # Phase 1: Many diverse greedy starts
    start_time = time.time()
    num_greedy = 30
    for i in range(num_greedy):
        if time.time() > deadline - 10:
            break
        adj, count = greedy()
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Phase 2: Degree-based greedy
    for i in range(10):
        if time.time() > deadline - 10:
            break
        adj, count = degree_greedy()
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Phase 3: Intensive local search on best
    if best_adj is not None:
        adj = best_adj[:]
        count = best_count
        
        # Multiple rounds of improvement
        for round_num in range(10):
            if time.time() > deadline - 5:
                break
            
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_swap(adj, count, max_iters=300)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_swap2(adj, count, max_iters=100)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 4: Genetic Algorithm
    if time.time() < deadline - 30:
        ga_adj, ga_count = genetic_algorithm(pop_size=18, generations=40)
        if ga_count > best_count:
            best_count = ga_count
            best_adj = ga_adj[:]
        
        # Polish the GA result
        if best_adj is not None:
            adj = best_adj[:]
            count = best_count
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            adj, count = local_swap(adj, count, max_iters=300)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            adj, count = local_swap2(adj, count, max_iters=150)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 5: Restart from best with perturbations
    if best_adj is not None and time.time() < deadline - 15:
        for i in range(20):
            if time.time() > deadline - 5:
                break
            adj = best_adj[:]
            count = best_count
            
            # Perturb: remove 2-5 random edges
            remove_count = random.randint(2, 5)
            for _ in range(remove_count):
                if count <= 0:
                    break
                u = random.randint(0, n - 1)
                mask = adj[u]
                if not mask:
                    continue
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                if v < u:
                    continue
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                count -= 1
            
            # Improve
            adj, count = local_add(adj, count)
            adj, count = local_swap(adj, count, max_iters=250)
            adj, count = local_add(adj, count)
            
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 6: Final intensive polish
    if best_adj is not None and time.time() < deadline - 3:
        adj = best_adj[:]
        count = best_count
        
        # Aggressive final add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final swaps
        adj, count = local_swap(adj, count, max_iters=600)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final swap2
        adj, count = local_swap2(adj, count, max_iters=200)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # One more final add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Convert to edge list
    if best_adj is None:
        return []
    
    return adj_to_edges(best_adj)
=======
    # Main search loop
    best_count = 0
    best_adj = None
    
    # Phase 1: Many diverse greedy starts
    start_time = time.time()
    num_greedy = 25
    for i in range(num_greedy):
        if time.time() > deadline - 15:
            break
        adj, count = greedy()
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Phase 2: Degree-based greedy
    for i in range(8):
        if time.time() > deadline - 15:
            break
        adj, count = degree_greedy()
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Phase 3: Intensive local search on best
    if best_adj is not None:
        adj = best_adj[:]
        count = best_count
        
        # Multiple rounds of improvement
        for round_num in range(12):
            if time.time() > deadline - 8:
                break
            
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_swap(adj, count, max_iters=400)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_swap2(adj, count, max_iters=120)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 4: Genetic Algorithm
    if time.time() < deadline - 35:
        ga_adj, ga_count = genetic_algorithm(pop_size=15, generations=35)
        if ga_count > best_count:
            best_count = ga_count
            best_adj = ga_adj[:]
        
        # Polish the GA result
        if best_adj is not None:
            adj = best_adj[:]
            count = best_count
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            adj, count = local_swap(adj, count, max_iters=350)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            adj, count = local_swap2(adj, count, max_iters=120)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 5: Restart from best with perturbations
    if best_adj is not None and time.time() < deadline - 12:
        for i in range(15):
            if time.time() > deadline - 5:
                break
            adj = best_adj[:]
            count = best_count
            
            # Perturb: remove 2-4 random edges
            remove_count = random.randint(2, 4)
            for _ in range(remove_count):
                if count <= 0:
                    break
                u = random.randint(0, n - 1)
                mask = adj[u]
                if not mask:
                    continue
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                if v < u:
                    continue
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                count -= 1
            
            # Improve
            adj, count = local_add(adj, count)
            adj, count = local_swap(adj, count, max_iters=300)
            adj, count = local_add(adj, count)
            
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 6: Final intensive polish
    if best_adj is not None and time.time() < deadline - 3:
        adj = best_adj[:]
        count = best_count
        
        # Aggressive final add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final swaps
        adj, count = local_swap(adj, count, max_iters=500)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final swap2
        adj, count = local_swap2(adj, count, max_iters=150)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # One more final add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final swap round
        adj, count = local_swap(adj, count, max_iters=300)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Last add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Convert to edge list
    if best_adj is None:
        return []
    
    return adj_to_edges(best_adj)
>>>>>>> REPLACE
