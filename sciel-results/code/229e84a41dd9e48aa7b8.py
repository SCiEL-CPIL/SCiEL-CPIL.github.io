import sys

def construct():
    def find_max_clique(vertices, local_adj):
        if not vertices:
            return []
        
        # Sort by degree in subgraph
        v_list = sorted(vertices, key=lambda x: len(local_adj.get(x, set())), reverse=True)
        
        best = []
        
        def backtrack(current, candidates):
            nonlocal best
            if len(current) > len(best):
                best = list(current)
            if not candidates:
                return
            if len(current) + len(candidates) <= len(best):
                return
            
            # Iterate from end to start (highest degree first)
            for i in range(len(candidates) - 1, -1, -1):
                v = candidates[i]
                # New candidates: those in candidates[0:i] connected to v
                new_candidates = [u for u in candidates[:i] if u in local_adj.get(v, set())]
                
                # Pruning
                if len(current) + 1 + len(new_candidates) <= len(best):
                    continue
                
                backtrack(current + [v], new_candidates)
                
        backtrack([], v_list)
        return best

    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        
        remaining_edges = set(edges)
        cliques = []
        
        while remaining_edges:
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            best_clique = []
            best_size = 0
            
            for v in range(n):
                if deg[v] == 0:
                    continue
                
                neighbors = []
                for u in range(n):
                    if u != v:
                        if (min(u, v), max(u, v)) in remaining_edges:
                            neighbors.append(u)
                
                if not neighbors:
                    continue
                
                local_adj = {u: set() for u in neighbors}
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        u, w = neighbors[i], neighbors[j]
                        if (min(u, w), max(u, w)) in remaining_edges:
                            local_adj[u].add(w)
                            local_adj[w].add(u)
                
                clique_in_neighbors = find_max_clique(neighbors, local_adj)
                candidate = clique_in_neighbors + [v]
                
                if len(candidate) > best_size:
                    best_size = len(candidate)
                    best_clique = candidate
            
            if not best_clique:
                e = next(iter(remaining_edges))
                best_clique = [e[0], e[1]]
            
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    u, v = best_clique[i], best_clique[j]
                    if u > v:
                        u, v = v, u
                    remaining_edges.discard((u, v))
            
            cliques.append(best_clique)
            
        return cliques

    res1 = instance(32, 0.6, 601)
    res2 = instance(64, 0.55, 602)
    
    return [res1, res2]
