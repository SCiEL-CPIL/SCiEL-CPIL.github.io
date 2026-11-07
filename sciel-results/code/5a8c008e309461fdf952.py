def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)
        
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        
        adj = [set() for _ in range(n)]
        internal_edges = set()
        
        for u, v in edges:
            if u < k and v < k:
                internal_edges.add((u, v))
            else:
                adj[v].add(u)
                
        i_vertices = list(range(k, n))
        i_vertices.sort(key=lambda x: len(adj[x]), reverse=True)
        
        cliques = []
        covered_internal_edges = set()
        
        for v in i_vertices:
            remaining_neighbors = list(adj[v])
            while remaining_neighbors:
                S = []
                for u in remaining_neighbors:
                    can_add = True
                    for w in S:
                        edge = (min(u, w), max(u, w))
                        if edge in covered_internal_edges:
                            can_add = False
                            break
                    if can_add:
                        S.append(u)
                        
                if not S:
                    S = [remaining_neighbors[0]]
                
                clique = [v] + S
                cliques.append(clique)
                
                for u1, u2 in itertools.combinations(S, 2):
                    covered_internal_edges.add((min(u1, u2), max(u1, u2)))
                
                S_set = set(S)
                remaining_neighbors = [u for u in remaining_neighbors if u not in S_set]
                
        remaining_internal = internal_edges - covered_internal_edges
        rem_adj = {u: set() for u in range(k)}
        for u, v in remaining_internal:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        while remaining_internal:
            u, v = next(iter(remaining_internal))
            clique = {u, v}
            candidates = rem_adj[u] & rem_adj[v]
            for w in candidates:
                if all(w in rem_adj[x] for x in clique):
                    clique.add(w)
                    
            cliques.append(list(clique))
            
            for a, b in itertools.combinations(clique, 2):
                remaining_internal.discard((min(a, b), max(a, b)))
                rem_adj[a].discard(b)
                rem_adj[b].discard(a)
                
        results.append(cliques)
        
    return results
