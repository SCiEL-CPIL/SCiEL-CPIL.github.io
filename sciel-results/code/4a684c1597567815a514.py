def construct():
    import time
    from random import Random
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges
    
    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # For each I-vertex, get its neighbors in K
        i_neighbors = {}
        for v in range(k, n):
            i_neighbors[v] = [u for u in range(k) if (u, v) in edges]
        
        used_k_edges = set()
        cliques = []
        covered_cross = set()
        
        # Pair up I-vertex neighbors to form triangles
        for v in range(k, n):
            if time.time() > deadline:
                break
            neighbors = i_neighbors[v][:]
            i = 0
            while i < len(neighbors) - 1:
                u1, u2 = neighbors[i], neighbors[i + 1]
                ke = (min(u1, u2), max(u1, u2))
                if ke not in used_k_edges:
                    cliques.append([int(u1), int(u2), int(v)])
                    used_k_edges.add(ke)
                    covered_cross.add((u1, v))
                    covered_cross.add((u2, v))
                    i += 2
                else:
                    i += 1
        
        # Cover remaining cross edges with 2-cliques
        for v in range(k, n):
            if time.time() > deadline:
                break
            for u in i_neighbors[v]:
                if (u, v) not in covered_cross:
                    cliques.append([int(u), int(v)])
        
        # Cover remaining K-edges
        remaining = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in used_k_edges:
                    remaining.add((u, w))
        
        while remaining:
            if time.time() > deadline:
                for (u, w) in remaining:
                    cliques.append([int(u), int(w)])
                break
            
            # Build adjacency
            adj = {u: set() for u in range(k)}
            for (u, w) in remaining:
                adj[u].add(w)
                adj[w].add(u)
            
            # Find max degree vertex
            start = max(adj, key=lambda x: len(adj[x]))
            clique = {start}
            candidates = list(adj[start])
            
            while candidates:
                best = None
                best_score = -1
                for c in candidates:
                    score = len(clique & adj[c])
                    if score > best_score:
                        best_score = score
                        best = c
                if best_score == 0:
                    break
                clique.add(best)
                candidates.remove(best)
            
            if len(clique) < 2:
                # Shouldn't happen, but fallback
                for (u, w) in remaining:
                    cliques.append([int(u), int(w)])
                remaining = set()
                break
            
            clique_list = sorted(clique)
            cliques.append([int(x) for x in clique_list])
            
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    e = (min(clique_list[i], clique_list[j]), max(clique_list[i], clique_list[j]))
                    remaining.discard(e)
        
        return cliques
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
        else:
            results.append(solve(n, k, p, seed))
    
    return results

