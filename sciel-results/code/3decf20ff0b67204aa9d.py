def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def greedy_maximal_clique(vertices, adj):
        if not vertices:
            return []
        
        v_set = set(vertices)
        sorted_v = sorted(vertices, key=lambda x: len(adj[x] & v_set), reverse=True)
        
        clique = []
        candidates = set(vertices)
        
        for v in sorted_v:
            if v not in candidates:
                continue
            is_connected = True
            for c in clique:
                if c not in adj[v]:
                    is_connected = False
                    break
            if is_connected:
                clique.append(v)
                candidates &= adj[v]
                
        return clique

    def split_graph_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        C = list(range(k))
        I = list(range(k, n))
        
        adj_C = {u: set() for u in C}
        adj_I = {v: set() for v in I}
        
        for u, v in edge_set:
            if u < k and v < k:
                adj_C[u].add(v)
                adj_C[v].add(u)
            elif u < k:
                adj_I[v].add(u)
            # else both in I, no edges

        cliques = []
        covered_C_edges = set()
        
        for v in I:
            N_v = adj_I[v]
            if N_v:
                clique = [v] + list(N_v)
                cliques.append(sorted(clique))
                for u in N_v:
                    for w in N_v:
                        if u < w:
                            covered_C_edges.add((u, w))
        
        remaining_C_edges = set()
        for u in C:
            for w in C:
                if u < w:
                    if (u, w) not in covered_C_edges:
                        remaining_C_edges.add((u, w))
                        
        if remaining_C_edges:
            remaining_set = set(remaining_C_edges)
            
            while remaining_set:
                active_vertices = set()
                for u, w in remaining_set:
                    active_vertices.add(u)
                    active_vertices.add(w)
                    
                local_adj = {u: set() for u in active_vertices}
                for u, w in remaining_set:
                    if u in local_adj and w in local_adj:
                        local_adj[u].add(w)
                        local_adj[w].add(u)
                        
                clique = greedy_maximal_clique(list(active_vertices), local_adj)
                
                if len(clique) >= 2:
                    cliques.append(sorted(clique))
                    to_remove = set()
                    for i in range(len(clique)):
                        for j in range(i+1, len(clique)):
                            u, w = clique[i], clique[j]
                            to_remove.add((min(u, w), max(u, w)))
                    remaining_set -= to_remove
                else:
                    u, w = next(iter(remaining_set))
                    cliques.append([u, w])
                    remaining_set.discard((u, w))

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = split_graph_partition(n, k, p, seed)
        results.append(cliques)
        
    return results
