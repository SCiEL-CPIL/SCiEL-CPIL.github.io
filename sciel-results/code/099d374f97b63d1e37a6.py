def construct():
    from random import Random
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
    
    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        # Build adjacency
        C = list(range(k))
        I = list(range(k, n))
        
        # cross-edges: (i, c) with i in I, c in C
        cross = {i: [] for i in I}
        for (a, b) in edges:
            if a < k and b < k:
                pass # C-edges
            elif a >= k and b >= k:
                pass # I-internal, shouldn't happen
            else:
                if a < k:
                    c, i = a, b
                else:
                    c, i = b, a
                cross[i].append(c)
        
        # all C-edges
        all_C_edges = set(frozenset((u, v)) for u in range(k) for v in range(u+1, k))
        remaining_C_edges = set(all_C_edges)
        
        mixed_cliques = []
        
        # Process I vertices
        # Sort by degree descending to handle large neighborhoods first? 
        # Or maybe ascending? 
        # Let's try descending to cover more C-edges early? 
        # Actually, if we cover C-edges early, later vertices might have to split more.
        # If we cover later, they have more freedom?
        # Let's stick to a deterministic order or degree-based.
        for i in sorted(I, key=lambda x: -len(cross[x])):
            neighbors = cross[i]
            if not neighbors:
                continue
            
            uncovered_neighbors = list(neighbors)
            
            while uncovered_neighbors:
                # Build graph of available edges among uncovered_neighbors
                # Edge (u,v) exists if frozenset((u,v)) is in remaining_C_edges
                # We want to find a maximal clique in this graph.
                
                # Helper to get neighbors in the available graph
                def get_avail_neighbors(v, candidates):
                    res = []
                    for u in candidates:
                        if u != v and frozenset((u, v)) in remaining_C_edges:
                            res.append(u)
                    return res
                
                # Greedy maximal clique
                # Start with a vertex with max degree in the available subgraph
                # To keep it simple and fast, just pick the first one?
                # Better: pick one with max degree to maximize size.
                
                # Calculate degrees in the induced subgraph on uncovered_neighbors
                degs = {}
                for u in uncovered_neighbors:
                    degs[u] = len(get_avail_neighbors(u, uncovered_neighbors))
                
                if not degs:
                    break
                    
                # Pick vertex with max degree
                start = max(degs, key=degs.get)
                
                clique = [start]
                clique_set = {start}
                candidates = [u for u in uncovered_neighbors if u != start]
                
                # Filter candidates to those connected to start
                candidates = [u for u in candidates if frozenset((u, start)) in remaining_C_edges]
                
                while candidates:
                    # Pick candidate with max degree in the current candidate set (among those connected to all in clique)
                    # Simplified: just pick first, or max degree in candidates
                    def deg_in_candidates(v, cands):
                        cnt = 0
                        for u in cands:
                            if u != v and frozenset((u, v)) in remaining_C_edges:
                                cnt += 1
                        return cnt
                        
                    # We need v to be connected to ALL in clique_set
                    # The candidates list already contains only those connected to all in clique_set?
                    # No, we only filtered for start initially.
                    # We need to ensure connectivity to all in clique.
                    
                    # Let's refine the loop:
                    # At each step, filter candidates to those connected to all in clique.
                    # Then pick one.
                    
                    # Actually, the standard greedy for maximal clique:
                    # 1. Start with empty set.
                    # 2. Add vertex with max degree.
                    # 3. Restrict to neighbors of all added vertices.
                    
                    # Let's restart the clique finding logic inside the while loop properly.
                    pass
                
                # Re-implementing the clique finding more cleanly
                clique = [start]
                clique_set = {start}
                candidates = [u for u in uncovered_neighbors if u != start]
                
                # Filter candidates to neighbors of start
                candidates = [u for u in candidates if frozenset((u, start)) in remaining_C_edges]
                
                while candidates:
                    # Find candidate with max degree in the subgraph induced by candidates
                    # That is also connected to all in clique (which is guaranteed if we filter correctly)
                    # But we need to filter candidates to be connected to ALL in clique.
                    
                    # Filter candidates
                    new_candidates = []
                    for u in candidates:
                        is_connected = True
                        for v in clique:
                            if frozenset((u, v)) not in remaining_C_edges:
                                is_connected = False
                                break
                        if is_connected:
                            new_candidates.append(u)
                    
                    if not new_candidates:
                        break
                        
                    # Pick one with max degree in new_candidates
                    def deg_in(v, cands):
                        cnt = 0
                        for u in cands:
                            if u != v and frozenset((u, v)) in remaining_C_edges:
                                cnt += 1
                        return cnt
                        
                    best_v = max(new_candidates, key=lambda v: deg_in(v, new_candidates))
                    clique.append(best_v)
                    clique_set.add(best_v)
                    candidates = new_candidates
                    # Remove best_v from candidates for next iteration?
                    # The filter `new_candidates` will handle it if we remove it?
                    # No, we just re-filter.
                    # But we need to exclude best_v from the next candidate pool?
                    # Yes, because we add it to clique.
                    candidates = [u for u in new_candidates if u != best_v]
                
                # Add the clique to mixed_cliques
                mixed_cliques.append([i] + clique)
                
                # Remove edges within clique from remaining_C_edges
                for idx_u in range(len(clique)):
                    for idx_v in range(idx_u + 1, len(clique)):
                        e = frozenset((clique[idx_u], clique[idx_v]))
                        remaining_C_edges.discard(e)
                
                # Remove clique vertices from uncovered_neighbors
                uncovered_neighbors = [u for u in uncovered_neighbors if u not in clique_set]
        
        # Cover remaining C-edges with pure cliques
        pure_cliques = []
        while remaining_C_edges:
            # Pick an arbitrary edge to start a clique
            # Find a maximal clique containing this edge
            edge = next(iter(remaining_C_edges))
            u, v = list(edge)
            
            clique = [u, v]
            clique_set = {u, v}
            candidates = [x for x in C if x != u and x != v]
            
            # Filter candidates to those connected to u and v
            candidates = [x for x in candidates if frozenset((x, u)) in remaining_C_edges and frozenset((x, v)) in remaining_C_edges]
            
            while candidates:
                # Filter to those connected to all in clique
                new_candidates = []
                for x in candidates:
                    is_connected = True
                    for y in clique:
                        if frozenset((x, y)) not in remaining_C_edges:
                            is_connected = False
                            break
                    if is_connected:
                        new_candidates.append(x)
                
                if not new_candidates:
                    break
                
                # Pick one
                # Heuristic: max degree in new_candidates
                def deg_in(v, cands):
                    cnt = 0
                    for u2 in cands:
                        if u2 != v and frozenset((u2, v)) in remaining_C_edges:
                            cnt += 1
                    return cnt
                    
                best_v = max(new_candidates, key=lambda v: deg_in(v, new_candidates))
                clique.append(best_v)
                clique_set.add(best_v)
                candidates = [x for x in new_candidates if x != best_v]
                
            pure_cliques.append(clique)
            
            # Remove edges within clique
            for idx_u in range(len(clique)):
                for idx_v in range(idx_u + 1, len(clique)):
                    e = frozenset((clique[idx_u], clique[idx_v]))
                    remaining_C_edges.discard(e)
        
        result = mixed_cliques + pure_cliques
        # Ensure types
        final_result = []
        for c in result:
            final_result.append(list(c))
        return final_result
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve(*spec))
    return results
