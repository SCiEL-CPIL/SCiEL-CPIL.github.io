def construct():
    from random import Random
    import heapq

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # C-C edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def get_maximal_clique_containing_edge(u, v, available_edges, all_vertices):
        """
        Finds a maximal clique in the subgraph defined by 'available_edges' 
        that must contain the edge (u, v).
        """
        if (u, v) not in available_edges:
            return None
            
        clique = {u, v}
        
        # Potential candidates are neighbors of both u and v in the available graph
        # Note: available_edges only contains uncovered edges.
        
        # Build adjacency for candidates
        # Candidates must be connected to u and v via available edges
        # And connected to each other via available edges if we want to add multiple.
        
        # Let's find all vertices w such that (min(u,w), max(u,w)) in available_edges 
        # and (min(v,w), max(v,w)) in available_edges.
        
        candidates = []
        for w in all_vertices:
            if w == u or w == v:
                continue
            e_uw = (min(u, w), max(u, w))
            e_vw = (min(v, w), max(v, w))
            if e_uw in available_edges and e_vw in available_edges:
                candidates.append(w)
        
        # Now we want to find a maximal clique in the subgraph induced by candidates 
        # plus u, v. 
        # Since u and v are already in, we just need to add vertices from candidates 
        # that are connected to all current clique members.
        
        # Sort candidates by degree in the induced subgraph of candidates + u + v?
        # Or just degree in candidates?
        # Let's compute degrees in the subgraph induced by candidates.
        
        # Actually, a simpler greedy:
        # While there exists a candidate w that is connected to ALL vertices in 'clique'
        # (using available_edges), add it.
        
        # To be efficient, we can pick the one with highest degree first.
        
        # Compute degree of each candidate in the subgraph induced by candidates.
        deg = {w: 0 for w in candidates}
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                w1, w2 = candidates[i], candidates[j]
                e = (min(w1, w2), max(w1, w2))
                if e in available_edges:
                    deg[w1] += 1
                    deg[w2] += 1
                    
        sorted_cands = sorted(candidates, key=lambda x: -deg[x])
        
        added = True
        while added:
            added = False
            for w in sorted_cands:
                if w in clique:
                    continue
                # Check if w is connected to all in clique
                is_connected = True
                for x in clique:
                    e = (min(w, x), max(w, x))
                    if e not in available_edges:
                        is_connected = False
                        break
                if is_connected:
                    clique.add(w)
                    added = True
                    # Update degrees? Not strictly necessary for this simple greedy, 
                    # but re-sorting might help. 
                    # For small k, this is fine.
        
        return list(clique)

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        all_vertices = set(range(n))
        remaining_edges = set(edges)
        
        cliques = []
        
        # To ensure deterministic behavior and avoid bias, 
        # we can process edges in a sorted order or pick the one with 
        # fewest options first (MRV heuristic).
        # Let's stick to a simple loop, but picking the edge that 
        # is in the fewest potential large cliques might be complex.
        # Just pick an arbitrary edge.
        
        # Optimization: If we pick an edge, we cover it.
        # We should prioritize edges that are "hard" to cover?
        # Edges in C-C are in many cliques. Edges in C-I are in fewer.
        # Let's just iterate through a sorted list of edges to be deterministic,
        # but skip those already covered.
        
        # Better: While remaining_edges is not empty:
        while remaining_edges:
            # Pick an edge. 
            # Heuristic: Pick the edge with the smallest number of 
            # available neighbors? 
            # For now, just pick the first one from a sorted list for stability.
            # Or, to be smarter, pick the edge that is part of the 
            # smallest maximal clique? 
            # Let's just pick any edge.
            e = next(iter(remaining_edges))
            u, v = e
            
            # Find a clique containing this edge in the remaining graph
            clique = get_maximal_clique_containing_edge(u, v, remaining_edges, all_vertices)
            
            if not clique:
                # Should not happen if e is in remaining_edges
                # Fallback: just the edge
                clique = [u, v]
            
            cliques.append(clique)
            
            # Remove all edges in this clique from remaining_edges
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    edge_to_remove = (min(a, b), max(a, b))
                    remaining_edges.discard(edge_to_remove)
        
        results.append(cliques)
    
    return results
