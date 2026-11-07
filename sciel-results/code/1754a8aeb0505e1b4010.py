def construct():
    from random import Random
    import itertools
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
        
    def min_clique_cover(edges_set, vertices):
        """
        Greedy minimum clique cover for a set of edges.
        Returns number of cliques.
        """
        if not edges_set:
            return 0
            
        remaining_edges = set(edges_set)
        num_cliques = 0
        
        while remaining_edges:
            # Find a maximal clique in the graph formed by remaining_edges
            # Vertices involved
            verts_in_edges = set()
            for u, v in remaining_edges:
                verts_in_edges.add(u)
                verts_in_edges.add(v)
                
            # Simple greedy: pick an edge, try to expand to clique
            # To be better, find max clique?
            # For k<=30, we can try to find a large clique.
            # Let's just pick the first edge and expand.
            # Actually, for minimization, we want LARGE cliques.
            
            # Heuristic: Find the vertex with highest degree in remaining_edges
            deg = {v: 0 for v in verts_in_edges}
            adj = {v: set() for v in verts_in_edges}
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
                adj[u].add(v)
                adj[v].add(u)
                
            # Pick vertex with max degree
            if not verts_in_edges:
                break
            start_v = max(verts_in_edges, key=lambda x: deg[x])
            
            # Greedy clique expansion
            clique = {start_v}
            candidates = set(adj[start_v])
            
            while candidates:
                # Pick candidate with max degree within candidates
                # To form a clique, all new vertices must be connected to all in clique
                # This is complex. Let's just use a simple maximal clique heuristic.
                # Actually, since the graph is a subgraph of a complete graph,
                # any subset of vertices is a clique IF all edges are present.
                # So we are looking for a clique in the graph (V, remaining_edges).
                
                # Simple approach: take start_v, add neighbors that are connected to all current clique members.
                # This finds a maximal clique.
                
                # Let's try to find the largest clique containing start_v.
                # Since k is small, we can iterate.
                # But this loop might be slow if done many times.
                
                # Alternative: Just use size-2 cliques?
                # That gives |R| cliques.
                # If we can find one big clique, we save a lot.
                
                # Let's try to find a clique of size at least 3.
                # If found, remove its edges.
                # Else, remove one edge.
                
                found_clique = False
                # Try to add vertices to clique
                current_clique = {start_v}
                # Sort candidates by degree
                sorted_cands = sorted(candidates, key=lambda x: deg[x], reverse=True)
                
                for v in sorted_cands:
                    # Check if v is connected to all in current_clique
                    is_connected = True
                    for u in current_clique:
                        if (min(u,v), max(u,v)) not in remaining_edges:
                            is_connected = False
                            break
                    if is_connected:
                        current_clique.add(v)
                        
                if len(current_clique) >= 2:
                    # Remove edges of this clique
                    for u, v in itertools.combinations(current_clique, 2):
                        e = (min(u,v), max(u,v))
                        if e in remaining_edges:
                            remaining_edges.remove(e)
                    num_cliques += 1
                else:
                    # Should not happen if remaining_edges not empty
                    break
                    
        return num_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        # Separate internal and cross edges
        internal_edges = set()
        cross_edges = []
        for u, v in edges:
            if u < k and v < k:
                internal_edges.add((u, v))
            else:
                cross_edges.append((u, v))
                
        # Build adjacency for I
        deg = [0] * n
        neighbors = [set() for _ in range(n)]
        for u, v in cross_edges:
            # u is in C, v is in I
            deg[v] += 1
            neighbors[v].add(u)
            
        # Sort I vertices by degree descending
        i_vertices = [i for i in range(k, n) if deg[i] > 0]
        i_vertices.sort(key=lambda x: deg[x], reverse=True)
        
        # Available internal edges
        avail_internal = set(internal_edges)
        
        cliques = []
        covered_cross = set()
        
        for i in i_vertices:
            N_i = neighbors[i]
            if not N_i:
                continue
                
            # Find largest subset S of N_i that is a clique in avail_internal
            # N_i is a set of vertices in C.
            # We want S such that all edges in S are in avail_internal.
            
            # Heuristic: Try to include all of N_i if possible.
            # Check if N_i is a clique in avail_internal
            is_clique = True
            for u, v in itertools.combinations(N_i, 2):
                e = (min(u, v), max(u, v))
                if e not in avail_internal:
                    is_clique = False
                    break
            
            if is_clique:
                S = N_i
            else:
                # Greedy: pick vertices from N_i one by one
                # Sort N_i by degree in avail_internal?
                # Just pick arbitrary?
                # Let's try to find a maximal clique in the graph induced by N_i with edges from avail_internal.
                S = set()
                # Simple greedy:
                # Sort N_i
                for v in N_i:
                    # Check if v can be added to S
                    # Must be connected to all in S via avail_internal
                    can_add = True
                    for u in S:
                        e = (min(u, v), max(u, v))
                        if e not in avail_internal:
                            can_add = False
                            break
                    if can_add:
                        S.add(v)
                        
            if len(S) >= 2:
                # Form clique S U {i}
                clique = list(S) + [i]
                cliques.append(clique)
                
                # Remove internal edges of S from avail_internal
                for u, v in itertools.combinations(S, 2):
                    e = (min(u, v), max(u, v))
                    if e in avail_internal:
                        avail_internal.remove(e)
                        
                # Mark cross edges as covered
                for u in S:
                    covered_cross.add((u, i))
            elif len(S) == 1:
                # Form size-2 clique
                u = list(S)[0]
                cliques.append([u, i])
                # No internal edges removed
                covered_cross.add((u, i))
            else:
                # S is empty? Should not happen if N_i not empty and we didn't filter.
                # But if all edges in N_i are used, S might be empty.
                # In that case, we must cover cross edges with size-2 cliques?
                # But we can't use size-2 if we don't have S?
                # Size-2 clique [u, i] does not consume internal edges.
                # So we should always be able to use size-2 cliques.
                # My greedy S selection might have failed to pick anything?
                # If S is empty, it means no vertex in N_i could be added?
                # That's impossible, we start with empty S.
                # First vertex always added.
                # So S has at least 1 vertex.
                pass

        # Cover remaining cross edges with size-2 cliques
        for u, v in cross_edges:
            if (u, v) not in covered_cross:
                cliques.append([u, v])
                
        # Cover remaining internal edges
        if avail_internal:
            # Use greedy clique cover
            # We just need the NUMBER? No, we need to return the cliques.
            # The function min_clique_cover above only returned count.
            # We need to generate the cliques.
            
            # Let's reuse the logic to generate cliques
            remaining_edges = set(avail_internal)
            while remaining_edges:
                verts_in_edges = set()
                for u, v in remaining_edges:
                    verts_in_edges.add(u)
                    verts_in_edges.add(v)
                    
                if not verts_in_edges:
                    break
                    
                deg = {v: 0 for v in verts_in_edges}
                adj = {v: set() for v in verts_in_edges}
                for u, v in remaining_edges:
                    deg[u] += 1
                    deg[v] += 1
                    adj[u].add(v)
                    adj[v].add(u)
                    
                start_v = max(verts_in_edges, key=lambda x: deg[x])
                
                current_clique = {start_v}
                candidates = set(adj[start_v])
                sorted_cands = sorted(candidates, key=lambda x: deg[x], reverse=True)
                
                for v in sorted_cands:
                    is_connected = True
                    for u in current_clique:
                        e = (min(u,v), max(u,v))
                        if e not in remaining_edges:
                            is_connected = False
                            break
                    if is_connected:
                        current_clique.add(v)
                        
                if len(current_clique) >= 2:
                    cliques.append(list(current_clique))
                    for u, v in itertools.combinations(current_clique, 2):
                        e = (min(u,v), max(u,v))
                        if e in remaining_edges:
                            remaining_edges.remove(e)
                else:
                    break
                    
        results.append(cliques)
        
    return results
