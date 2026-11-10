def construct():
    import time
    start_time = time.time()
    DEADLINE = 145
    
    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)
    
    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        cross_neighbors = {}
        for v in range(k, n):
            neighbors = [u for u in range(k) if (u, v) in edge_set]
            if neighbors:
                cross_neighbors[v] = neighbors
        
        cliques = []
        covered_internal = set()
        
        for v in sorted(cross_neighbors.keys()):
            neighbors = cross_neighbors[v]
            clique = [int(v)] + [int(u) for u in neighbors]
            cliques.append(clique)
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a, b = int(neighbors[i]), int(neighbors[j])
                    covered_internal.add((min(a, b), max(a, b)))
        
        remaining = set()
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in covered_internal:
                    remaining.add((u, v))
        
        while remaining:
            u, v = next(iter(remaining))
            clique_verts = {u, v}
            changed = True
            while changed:
                changed = False
                for w in range(k):
                    if w not in clique_verts:
                        ok = True
                        for x in clique_verts:
                            e = (min(w, x), max(w, x))
                            if e not in remaining:
                                ok = False
                                break
                        if ok:
                            clique_verts.add(w)
                            changed = True
            
            verts = sorted(clique_verts)
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    remaining.discard((verts[i], verts[j]))
            
            cliques.append([int(x) for x in verts])
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() - start_time > DEADLINE:
            break
        cliques = solve_one(*spec)
        results.append(cliques)
    
    return results
