def construct():
    import time
    import random
    
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj
    
    def min_vertex_cover(adj):
        n = len(adj)
        # Build edge list
        edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.add((u, v))
        
        if not edges:
            return set()
        
        # Greedy vertex cover
        cover = set()
        remaining_edges = set(edges)
        deg = [len(adj[i]) for i in range(n)]
        
        while remaining_edges:
            # Pick vertex with max degree in remaining graph
            best_v = -1
            best_deg = -1
            for v in range(n):
                if v in cover:
                    continue
                # Count remaining edges incident to v
                d = 0
                for u in adj[v]:
                    if (min(v,u), max(v,u)) in remaining_edges:
                        d += 1
                if d > best_deg:
                    best_deg = d
                    best_v = v
            if best_v == -1 or best_deg == 0:
                break
            cover.add(best_v)
            # Remove all edges incident to best_v
            to_remove = set()
            for u in adj[best_v]:
                e = (min(best_v, u), max(best_v, u))
                if e in remaining_edges:
                    to_remove.add(e)
            remaining_edges -= to_remove
        
        # Reduction: try to remove vertices from cover
        changed = True
        while changed:
            changed = False
            for v in list(cover):
                # Try removing v
                candidate = cover - {v}
                # Check if candidate is still a vertex cover
                valid = True
                for (u, w) in edges:
                    if u not in candidate and w not in candidate:
                        valid = False
                        break
                if valid:
                    cover = candidate
                    changed = True
                    if time.time() > deadline - 5:
                        break
            if time.time() > deadline - 5:
                break
        
        return cover
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 10:
            # Not enough time, return empty (invalid but better than timeout)
            out.append([])
            continue
        adj = random_graph(n, p, seed)
        cover = min_vertex_cover(adj)
        out.append(sorted(cover))
    
    return out
