    def greedy_decompose(edges, adj, strategy='max'):
        rem_edges = set(edges)
        # degrees
        deg = {v: 0 for v in adj}
        for u, v in rem_edges:
            deg[u] += 1
            deg[v] += 1
            
        paths = []
        while rem_edges:
            # Start vertex: pick one with max degree
            start = max(deg.keys(), key=lambda x: deg[x])
            
            path = [start]
            visited = {start}
            current = start
            
            while True:
                # Find candidates
                candidates = []
                for nb in adj[current]:
                    key = (min(current, nb), max(current, nb))
                    if key in rem_edges and nb not in visited:
                        candidates.append(nb)
                        
                if not candidates:
                    break
                    
                # Select next
                if strategy == 'max':
                    next_v = max(candidates, key=lambda x: deg[x])
                elif strategy == 'min':
                    next_v = min(candidates, key=lambda x: deg[x])
                else:
                    # Random
                    next_v = random.choice(candidates)
                    
                key = (min(current, next_v), max(current, next_v))
                rem_edges.discard(key)
                deg[current] -= 1
                deg[next_v] -= 1
                visited.add(next_v)
                path.append(next_v)
                current = next_v
                
            paths.append(path)
            
        return paths
    