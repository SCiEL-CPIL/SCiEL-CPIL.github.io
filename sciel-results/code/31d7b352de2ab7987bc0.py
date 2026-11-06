def construct():
    import random
    from functools import lru_cache

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        # Generate edges
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            return []

        # Map edge to index
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # Build adjacency list for original graph
        adj_orig = [[] for _ in range(n)]
        for u, v in edges:
            adj_orig[u].append(v)
            adj_orig[v].append(u)
        
        # Build conflict graph adjacency bitmasks
        # Conflict if:
        # 1. Share a vertex
        # 2. Are joined by an edge (induced matching constraint)
        
        adj_conf = [0] * m
        
        for i in range(m):
            u1, v1 = edges[i]
            # Mark vertices involved
            verts_i = {u1, v1}
            # Neighbors of u1 and v1
            neigh_i = set(adj_orig[u1]) | set(adj_orig[v1])
            
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                
                # Check 1: Share vertex
                if (u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2):
                    # Conflict
                    adj_conf[i] |= (1 << j)
                    adj_conf[j] |= (1 << i)
                    continue
                
                # Check 2: Joined by an edge
                # Edge (u1, v1) and (u2, v2) are joined if there is an edge in G
                # connecting a vertex in {u1, v1} to a vertex in {u2, v2}
                # Since they don't share vertices, we check:
                # (u1, u2) in G? (u1, v2) in G? (v1, u2) in G? (v1, v2) in G?
                
                # We can use the adjacency sets for O(1) check
                adj_set_u1 = set(adj_orig[u1])
                adj_set_v1 = set(adj_orig[v1])
                
                # Actually, building sets for every i,j is expensive.
                # Let's precompute adjacency sets for all vertices.
                
        # Optimization: Precompute adjacency sets
        adj_sets = [set(adj_orig[u]) for u in range(n)]
        
        # Re-initialize adj_conf
        adj_conf = [0] * m
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                
                # Check shared vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    adj_conf[i] |= (1 << j)
                    adj_conf[j] |= (1 << i)
                else:
                    # Check if joined by edge
                    # Check (u1, u2), (u1, v2), (v1, u2), (v1, v2)
                    if u2 in adj_sets[u1] or v2 in adj_sets[u1] or u2 in adj_sets[v1] or v2 in adj_sets[v1]:
                        adj_conf[i] |= (1 << j)
                        adj_conf[j] |= (1 << i)

        # MIS Solver on Conflict Graph
        # We want to find Max Independent Set in G_conf.
        # Independent Set in G_conf = Induced Matching in G_orig.
        
        best_mis_size = 0
        best_mis_mask = 0
        
        # Greedy Lower Bound for initial best
        def greedy_mis(candidates):
            size = 0
            mask = 0
            while candidates:
                # Pick node with min degree in remaining subgraph? Or just first?
                # Let's pick the first available node for simplicity in heuristic
                # But better to pick one that allows more future nodes.
                # For speed, just pick lowest bit.
                v = (candidates & -candidates).bit_length() - 1
                mask |= (1 << v)
                size += 1
                # Remove v and its neighbors
                candidates &= ~(adj_conf[v] | (1 << v))
            return size, mask

        # Initial greedy
        lb_size, lb_mask = greedy_mis((1 << m) - 1)
        best_mis_size = lb_size
        best_mis_mask = lb_mask

        # Branch and Bound
        # To avoid recursion depth issues and speed up, we can use iterative deepening or just standard BB.
        # Given m <= 240, recursion depth is manageable if we prune well.
        
        import sys
        sys.setrecursionlimit(10000)

        def bb_search(candidates, current_mask, current_size):
            nonlocal best_mis_size, best_mis_mask
            
            # Pruning
            if current_size + candidates.bit_count() <= best_mis_size:
                return
            
            if candidates == 0:
                if current_size > best_mis_size:
                    best_mis_size = current_size
                    best_mis_mask = current_mask
                return
            
            # Heuristic: Get a quick lower bound from greedy on remaining candidates
            # This helps prune earlier
            g_size, g_mask = greedy_mis(candidates)
            if current_size + g_size <= best_mis_size:
                return
            
            # Select pivot
            # Choose a node from candidates.
            # Heuristic: Pick node with highest degree in the induced subgraph?
            # Or just the first one. Let's pick the one with the smallest index for determinism.
            # A better heuristic is to pick a node that is part of a triangle or high degree.
            # For now, let's pick the node with the highest degree in the remaining graph.
            # Calculating degree for all candidates might be expensive.
            # Let's just pick the first bit.
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            
            # Branch 1: Include v
            # New candidates: candidates without v and without neighbors of v
            new_candidates = candidates & ~(adj_conf[v] | v_bit)
            bb_search(new_candidates, current_mask | v_bit, current_size + 1)
            
            # Branch 2: Exclude v
            new_candidates = candidates & ~v_bit
            bb_search(new_candidates, current_mask, current_size)

        bb_search((1 << m) - 1, 0, 0)
        
        # Now we have the largest induced matching (MIS of conflict graph).
        # We assign color 0 to these edges.
        # Then remove them and repeat.
        
        colors = [0] * m
        available_mask = (1 << m) - 1
        color_idx = 0
        
        while available_mask:
            # Find MIS in subgraph induced by available_mask
            # We need to modify the BB search to work on a subset
            # The adj_conf is for the whole graph.
            # The candidates passed to bb_search are the available nodes.
            # The pruning and logic remain the same.
            
            local_best_size = 0
            local_best_mask = 0
            
            # Reset global bests for this sub-instance
            # We can just run the same BB logic but constrained to available_mask
            
            def bb_search_subset(candidates, current_mask, current_size):
                nonlocal local_best_size, local_best_mask
                
                if current_size + candidates.bit_count() <= local_best_size:
                    return
                
                if candidates == 0:
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_mask = current_mask
                    return
                
                # Greedy bound
                g_size, g_mask = greedy_mis(candidates)
                if current_size + g_size <= local_best_size:
                    return
                
                v_bit = candidates & -candidates
                v = v_bit.bit_length() - 1
                
                # Include v
                new_candidates = candidates & ~(adj_conf[v] | v_bit)
                bb_search_subset(new_candidates, current_mask | v_bit, current_size + 1)
                
                # Exclude v
                new_candidates = candidates & ~v_bit
                bb_search_subset(new_candidates, current_mask, current_size)

            bb_search_subset(available_mask, 0, 0)
            
            # Assign color
            mis_mask = local_best_mask
            # Iterate through bits in mis_mask
            temp_mask = mis_mask
            while temp_mask:
                bit = temp_mask & -temp_mask
                idx = bit.bit_length() - 1
                colors[idx] = color_idx
                available_mask &= ~bit
                temp_mask &= ~bit
            
            color_idx += 1
            
        return colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        res = solve_instance(n, d, seed)
        out.append(res)
        
    return out
