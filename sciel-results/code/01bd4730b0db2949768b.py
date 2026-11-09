import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_paths(edges):
        # edges is a list of tuples
        # Build adjacency list
        n = max(max(e) for e in edges) + 1 if edges else 0
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        
        paths = []
        remaining_edges = set(edge_set)
        
        # Heuristic: While edges remain, find a long simple path
        while remaining_edges:
            # Pick a start vertex: one with max degree in remaining graph
            # Degrees in remaining graph
            deg = {i: 0 for i in range(n)}
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            # Find vertex with max degree
            if not any(d > 0 for d in deg.values()):
                break
                
            start = max(deg, key=lambda x: deg[x])
            
            path = [start]
            visited = {start}
            current = start
            
            while True:
                # Find neighbors of current in remaining_edges not in visited
                candidates = []
                for v in adj[current]:
                    if v in visited:
                        continue
                    # Check if edge (current, v) is in remaining_edges
                    e1 = (current, v)
                    e2 = (v, current)
                    if e1 in remaining_edges or e2 in remaining_edges:
                        # Heuristic: prefer neighbor with higher remaining degree
                        candidates.append((deg[v], v))
                
                if not candidates:
                    break
                
                # Pick the best candidate (highest degree)
                candidates.sort(key=lambda x: x[0], reverse=True)
                next_v = candidates[0][1]
                
                # Remove edge from remaining
                e1 = (current, next_v)
                e2 = (next_v, current)
                if e1 in remaining_edges:
                    remaining_edges.remove(e1)
                elif e2 in remaining_edges:
                    remaining_edges.remove(e2)
                    
                path.append(next_v)
                visited.add(next_v)
                current = next_v
                
            paths.append(path)
            
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return seed-like result (each edge a path) for remaining
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        paths = decompose_paths(edges)
        results.append(paths)

    return results
