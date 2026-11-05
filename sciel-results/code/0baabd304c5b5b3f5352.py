def construct():
    from random import Random
    import itertools
    
    def instance(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges
    
    def greedy_clique_partition(edges):
        if not edges:
            return []
        
        n = max(max(u for u, v in edges), max(v for u, v in edges)) + 1
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        unassigned = set(edges)
        cliques = []
        
        while unassigned:
            u, v = next(iter(unassigned))
            
            # Build initial candidate set: common neighbors of u and v in unassigned graph
            # w such that (u,w) and (v,w) are in unassigned
            candidates = set()
            for w in adj[u]:
                if w == v:
                    continue
                eu_w = (min(u, w), max(u, w))
                ev_w = (min(v, w), max(v, w))
                if eu_w in unassigned and ev_w in unassigned:
                    candidates.add(w)
            
            clique = {u, v}
            
            # Greedily add candidates
            changed = True
            while changed:
                changed = False
                for w in list(candidates):
                    can_add = True
                    for x in clique:
                        if w == x:
                            continue
                        a, b = min(w, x), max(w, x)
                        if (a, b) not in unassigned:
                            can_add = False
                            break
                    if can_add:
                        clique.add(w)
                        candidates.discard(w)
                        changed = True
            
            # Collect all edges in clique
            clique_vertices = sorted(clique)
            clique_edges = set()
            valid = True
            for i in range(len(clique_vertices)):
                for j in range(i + 1, len(clique_vertices)):
                    a, b = clique_vertices[i], clique_vertices[j]
                    e = (a, b)
                    if e not in unassigned:
                        valid = False
                        break
                    clique_edges.add(e)
                if not valid:
                    break
            
            if not valid or not clique_edges:
                # Fallback: single edge
                clique_edges = {(u, v)}
                clique_vertices = [u, v]
            
            cliques.append(clique_vertices)
            unassigned -= clique_edges
        
        return cliques
    
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = instance(n, p, seed)
        cliques = greedy_clique_partition(edges)
        result.append(cliques)
    
    return result
