<<<<<<< SEARCH
    # Now try random search for larger graphs
    # Strategy: Try various n from 40 down to (best_result['n']+1 if best_result else 28)
    min_target = (best_result['n'] + 1) if best_result else 28
    
    for target_n in range(40, min_target - 1, -1):
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 5000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Try different random generation strategies
            strategy = random.randint(0, 2)
            
            if strategy == 0:
                # Random edge addition (original approach)
                adj = make_random_triangle_free(target_n)
            elif strategy == 1:
                # Random bipartite with some added edges
                adj = random_bipartite(target_n, 0.5)
                # Try to add some edges within parts while maintaining triangle-free
                half = target_n // 2
                extra_edges = [(i, j) for i in range(half) for j in range(i + 1, half)]
                extra_edges += [(i, j) for i in range(half, target_n) for j in range(i + 1, target_n)]
                random.shuffle(extra_edges)
                for u, v in extra_edges:
                    common = adj[u] & adj[v]
                    if common == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            else:
                # Blow up a small triangle-free graph
                # Use C5 blown up
                base_n = 5
                base_adj = adj_c5[:]
                # Random sizes that sum to target_n
                sizes = [1] * base_n
                remaining = target_n - base_n
                for _ in range(remaining):
                    sizes[random.randint(0, base_n - 1)] += 1
                adj, _ = blow_up(base_adj, base_n, sizes)
            
            if not has_independent_set_10(adj, target_n):
                found = True
                best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                break
        
        if found:
            break
=======
    # Try systematic extension of Andrásfai graph
    # And(9) has n=27, alpha=9. Try to add vertices one by one.
    adj9, n9 = make_andrasfai(9)
    if not has_independent_set_10(adj9, n9):
        if best_result is None or n9 > best_result['n']:
            best_result = {'n': n9, 'edges': build_edge_list(adj9, n9)}
    
    # Try to extend And(9) by adding vertices
    current_adj = adj9[:]
    current_n = n9
    for extra in range(1, 14):  # Try to add up to 13 more vertices
        if time.time() > deadline:
            break
        extended = extend_andrasfai(current_adj, current_n, 1)
        if extended is not None:
            ext_adj, ext_n = extended
            if not has_independent_set_10(ext_adj, ext_n) and is_triangle_free(ext_adj, ext_n):
                current_adj = ext_adj
                current_n = ext_n
                if best_result is None or current_n > best_result['n']:
                    best_result = {'n': current_n, 'edges': build_edge_list(current_adj, current_n)}
    
    # Now try random search for larger graphs
    # Strategy: Try various n from 40 down to (best_result['n']+1 if best_result else 28)
    min_target = (best_result['n'] + 1) if best_result else 28
    
    for target_n in range(40, min_target - 1, -1):
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 8000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Try different random generation strategies
            strategy = random.randint(0, 3)
            
            if strategy == 0:
                # Random edge addition (original approach)
                adj = make_random_triangle_free(target_n)
            elif strategy == 1:
                # Random bipartite with some added edges
                adj = random_bipartite(target_n, 0.5)
                # Try to add some edges within parts while maintaining triangle-free
                half = target_n // 2
                extra_edges = [(i, j) for i in range(half) for j in range(i + 1, half)]
                extra_edges += [(i, j) for i in range(half, target_n) for j in range(i + 1, target_n)]
                random.shuffle(extra_edges)
                for u, v in extra_edges:
                    common = adj[u] & adj[v]
                    if common == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            elif strategy == 2:
                # Blow up a small triangle-free graph
                # Use C5 blown up
                base_n = 5
                base_adj = adj_c5[:]
                # Random sizes that sum to target_n
                sizes = [1] * base_n
                remaining = target_n - base_n
                for _ in range(remaining):
                    sizes[random.randint(0, base_n - 1)] += 1
                adj, _ = blow_up(base_adj, base_n, sizes)
            else:
                # Start from Andrásfai-like structure and randomize
                # Create a cyclic-like graph with random perturbations
                adj = [0] * target_n
                # First create a sparse cyclic backbone
                for i in range(target_n):
                    for offset in [1, 2, target_n - 1, target_n - 2]:
                        j = (i + offset) % target_n
                        if j > i:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                # Then add random edges while maintaining triangle-freeness
                all_edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n) 
                             if not (adj[i] & (1 << j))]
                random.shuffle(all_edges)
                for u, v in all_edges:
                    common = adj[u] & adj[v]
                    if common == 0:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            if not has_independent_set_10(adj, target_n):
                found = True
                best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                break
        
        if found:
            break
>>>>>>> REPLACE

