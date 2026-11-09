def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def split_graph_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        C = list(range(k))
        I = list(range(k, n))
        
        # Build neighborhoods for I vertices
        # N[v] is the set of neighbors in C
        N = {v: set() for v in I}
        for u, w in edge_set:
            if u in N: # u is in I
                N[u].add(w)
            elif w in N: # w is in I
                N[w].add(u)
                
        # Sort I vertices by neighborhood size descending
        I_sorted = sorted(I, key=lambda v: len(N[v]), reverse=True)
        
        cliques = []
        
        # remaining_C_edges: set of edges within C that are not yet covered
        # Initialize with all possible edges in C
        remaining_C_edges = set((u, w) for u in C for w in C if u < w)
        
        # Process each I vertex
        for v in I_sorted:
            # The neighbors of v in C that are still "active"
            # Actually, the edge (v, u) is unique to v. So if (v, u) is in edge_set, it must be covered by a clique containing v.
            # Since v is in I, it can only be in cliques with C vertices.
            # We can cover all edges (v, u) for u in N[v] in a single clique {v} U S, where S is a subset of N[v].
            # To maximize coverage of C-C edges, we should pick S as large as possible?
            # If we pick S = N[v], we cover all edges (u, w) for u, w in N[v].
            # Is there a reason not to pick S = N[v]?
            # The clique is {v} U N[v].
            # This covers all edges incident to v.
            # And covers all edges between pairs in N[v].
            # This seems optimal for v.
            
            S = N[v]
            if not S:
                continue
                
            clique = [v] + list(S)
            cliques.append(sorted(clique))
            
            # Remove edges covered by this clique from remaining_C_edges
            # Edges covered are (u, w) for u, w in S
            for u in S:
                for w in S:
                    if u < w:
                        remaining_C_edges.discard((u, w))
            
        # Now, partition the remaining_C_edges into cliques
        # The vertices involved are a subset of C.
        # We can use a greedy algorithm to find maximal cliques in the graph formed by remaining_C_edges.
        
        if remaining_C_edges:
            # Build adjacency list for the remaining graph
            adj = {u: set() for u in C}
            for u, w in remaining_C_edges:
                adj[u].add(w)
                adj[w].add(u)
                
            remaining = set(remaining_C_edges)
            
            while remaining:
                # Find a vertex with max degree in remaining
                deg = {u: 0 for u in C}
                for u, w in remaining:
                    deg[u] += 1
                    deg[w] += 1
                    
                start = max(deg, key=lambda x: deg[x])
                
                # Greedily build a clique containing start
                clique = [start]
                candidates = list(adj[start])
                # Sort candidates by degree in remaining
                candidates.sort(key=lambda x: deg[x], reverse=True)
                
                for c in candidates:
                    # Check if c is connected to all in clique
                    connected = True
                    for m in clique:
                        if (min(c, m), max(c, m)) not in remaining:
                            connected = False
                            break
                    if connected:
                        clique.append(c)
                        
                if len(clique) >= 2:
                    cliques.append(sorted(clique))
                    # Remove edges
                    to_remove = set()
                    for i in range(len(clique)):
                        for j in range(i+1, len(clique)):
                            u, w = clique[i], clique[j]
                            to_remove.add((min(u, w), max(u, w)))
                    remaining -= to_remove
                else:
                    # Fallback: take one edge
                    u, w = next(iter(remaining))
                    cliques.append([u, w])
                    remaining.discard((min(u, w), max(u, w)))
                    
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = split_graph_partition(n, k, p, seed)
        results.append(cliques)
        
    return results
