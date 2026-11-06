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
    # Now try random search for larger graphs
    # Strategy: Try various n from 40 down to (best_result['n']+1 if best_result else 28)
    min_target = (best_result['n'] + 1) if best_result else 28
    
    for target_n in range(40, min_target - 1, -1):
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 3000
        
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
            
            # Check if already valid
            if not has_independent_set_10(adj, target_n):
                found = True
                best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                break
            
            # Try to improve by adding edges to reduce alpha
            if try_add_edges_to_reduce_alpha(adj, target_n, max_rounds=50):
                found = True
                best_result = {'n': target_n, 'edges': build_edge_list(adj, target_n)}
                break
        
        if found:
            break
>>>>>>> REPLACE

