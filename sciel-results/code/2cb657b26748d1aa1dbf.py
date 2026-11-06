def construct():
    import time
    import random
    
    DEADLINE = 140
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = adj[v][u] = True
                    edges.append((u, v))
        return adj, edges
    
    def solve_instance(n, p, seed, time_limit):
        adj, edges = random_graph(n, p, seed)
        edge_set = set(edges)
        cliques = []
        start_time = time.time()
        
        # Adjacency sets for fast lookup
        adj_set = [set() for _ in range(n)]
        for u, v in edges:
            adj_set[u].add(v)
            adj_set[v].add(u)
        
        while edge_set:
            if time.time() - start_time > time_limit:
                break
            
            # Find best clique greedily
            # Try starting from high-degree vertices in remaining graph
            deg = [0] * n
            for e in edge_set:
                deg[e[0]] += 1
                deg[e[1]] += 1
            
            vertices = [v for v in range(n) if deg[v] > 0]
            vertices.sort(key=lambda v: deg[v], reverse=True)
            
            best_clique = None
            best_edge_count = 0
            
            # Try limited number of starts
            for start in vertices[:20]:
                # Greedy clique: start with start, add neighbors in order of connectivity
                clique = [start]
                in_clique = {start}
                
                # Candidates: neighbors of start with edges to start in edge_set
                cands = [v for v in adj_set[start] if (start, v) in edge_set or (v, start) in edge_set]
                
                # Sort by degree in remaining graph
                cands.sort(key=lambda v: deg[v], reverse=True)
                
                for v in cands:
                    if all((v, u) in edge_set or (u, v) in edge_set for u in in_clique):
                        clique.append(v)
                        in_clique.add(v)
                
                # Count edges covered
                cnt = 0
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        if (a, b) in edge_set or (b, a) in edge_set:
                            cnt += 1
                
                if cnt > best_edge_count:
                    best_edge_count = cnt
                    best_clique = clique
            
            if not best_clique or best_edge_count == 0:
                e = next(iter(edge_set))
                best_clique = list(e)
            
            # Remove edges
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    a, b = best_clique[i], best_clique[j]
                    if (a, b) in edge_set:
                        edge_set.remove((a, b))
                    elif (b, a) in edge_set:
                        edge_set.remove((b, a))
            
            cliques.append(best_clique)
        
        # Fix any remaining edges
        covered = set()
        for c in cliques:
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    covered.add(frozenset((c[i], c[j])))
        for e in edges:
            if frozenset(e) not in covered:
                cliques.append(list(e))
        
        return cliques
    
    results = []
    t0 = time.time()
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        rem = DEADLINE - (time.time() - t0)
        if rem < 5:
            adj, edges = random_graph(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        tl = min(60, rem - 5)
        results.append(solve_instance(n, p, seed, tl))
    
    return results
