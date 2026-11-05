def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    results = []
    
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            results.append([[list(e)] for e in instance(*spec)])
            continue
            
        n = spec[0]
        edges = set(instance(*spec))
        
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        pieces = []
        
        while edges:
            if time.time() > deadline:
                for u, v in edges:
                    pieces.append([u, v])
                break

            # Find a cycle
            # Pick a start node with high degree
            start_node = max(edges.union(set([u for u, v in edges])), key=lambda x: len(adj[x]))
            
            visited = set()
            path = []
            path_idx = {}
            best_cycle = []
            
            def dfs(u, parent):
                nonlocal best_cycle
                visited.add(u)
                path.append(u)
                path_idx[u] = len(path) - 1
                
                # Sort neighbors by degree to find long cycles faster?
                # High degree neighbors are more likely to lead to long paths
                neighbors = sorted(adj[u], key=lambda x: -len(adj[x]))
                
                for v in neighbors:
                    if v == parent:
                        continue
                    if v in path_idx:
                        # Cycle found
                        start_idx = path_idx[v]
                        cycle = path[start_idx:] + [v]
                        if len(cycle) > len(best_cycle):
                            best_cycle = cycle[:]
                    elif v not in visited:
                        dfs(v, u)
                        
                path.pop()
                del path_idx[u]
                # Do not remove from visited, as we want to explore the whole component
                # to find the longest fundamental cycle.
                
            if adj[start_node]:
                dfs(start_node, -1)
            
            if best_cycle:
                # Remove edges of best_cycle
                for i in range(len(best_cycle) - 1):
                    u, v = best_cycle[i], best_cycle[i+1]
                    e = tuple(sorted((u, v)))
                    edges.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
                pieces.append(best_cycle)
            else:
                # No cycle, take an edge
                u, v = next(iter(edges))
                edges.remove((u, v))
                adj[u].remove(v)
                adj[v].remove(u)
                pieces.append([u, v])
                
        results.append(pieces)

    return results
