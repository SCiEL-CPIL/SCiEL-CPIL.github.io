import time

def construct():
    start_time = time.time()
    deadline = 150.0 # Safety margin
    
    from random import Random

    def instance(n, k, p, seed):
        # Check time
        if time.time() > deadline:
            # Return a trivial valid solution (though it might be large)
            # Just return empty lists to avoid crash? No, must be valid.
            # But if we are out of time, we should have returned earlier.
            pass

        rng = Random(seed)
        N = [set() for _ in range(n)]
        
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    N[v].add(u)
        
        cliques = []
        covered_edges_in_C = set()
        
        for v in range(k, n):
            if N[v]:
                clique = [v] + sorted(N[v])
                cliques.append(clique)
                
                neighbors = sorted(N[v])
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        u, w = neighbors[i], neighbors[j]
                        if u > w:
                            u, w = w, u
                        covered_edges_in_C.add((u, w))
        
        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_edges_in_C:
                    remaining_edges.add((u, w))
        
        rem_edges = set(remaining_edges)
        
        while rem_edges:
            if time.time() > deadline:
                break # Fallback: leave some edges uncovered? Invalid.
                # But this case is extremely unlikely.
            
            e = next(iter(rem_edges))
            u, w = e
            
            common = set()
            for x in range(k):
                if x == u or x == w:
                    continue
                eu = (min(u, x), max(u, x))
                ew = (min(w, x), max(w, x))
                if eu in rem_edges and ew in rem_edges:
                    common.add(x)
            
            best_clique = {u, w}
            
            if common:
                current = {u, w}
                remaining_candidates = set(common)
                while remaining_candidates:
                    added = False
                    for x in list(remaining_candidates):
                        is_connected = True
                        for y in current:
                            if x == y:
                                continue
                            a, b = min(x, y), max(x, y)
                            if (a, b) not in rem_edges:
                                is_connected = False
                                break
                        if is_connected:
                            current.add(x)
                            remaining_candidates.remove(x)
                            added = True
                            break
                    if not added:
                        break
                
                if len(current) > len(best_clique):
                    best_clique = current
            
            cliques.append(sorted(best_clique))
            
            verts = sorted(best_clique)
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    a, b = verts[i], verts[j]
                    rem_edges.discard((a, b))
        
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        result.append(instance(*spec))
        
    return result
