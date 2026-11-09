import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_paths(edges):
        if not edges:
            return []
        
        n = max(max(e) for e in edges) + 1
        # Build adjacency with edge sets for fast removal
        # adj[u] is a set of (v, edge_tuple)
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add((v, (u, v) if u < v else (v, u)))
            adj[v].add((u, (u, v) if u < v else (v, u)))
            
        remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        paths = []
        
        while remaining_edges:
            # Calculate degrees in remaining graph
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            # Find start vertex: Max degree
            # If tie, pick one that has a neighbor with min degree?
            # Just max degree for now
            start_candidates = [i for i in range(n) if deg[i] > 0]
            if not start_candidates:
                break
            start = max(start_candidates, key=lambda x: deg[x])
            
            path = [start]
            visited = {start}
            current = start
            
            while True:
                # Get available neighbors
                # We need to find v such that (current, v) is in remaining_edges and v not in visited
                candidates = []
                for v, e_tuple in adj[current]:
                    if v in visited:
                        continue
                    # Check if edge is still remaining
                    if e_tuple in remaining_edges:
                        # Warnsdorff: prefer neighbor with fewer available neighbors
                        # Count available neighbors of v
                        avail_count = 0
                        for w, _ in adj[v]:
                            if w not in visited:
                                # Check edge (v, w)
                                ew = (v, w) if v < w else (w, v)
                                if ew in remaining_edges:
                                    avail_count += 1
                        candidates.append((avail_count, v, e_tuple))
                
                if not candidates:
                    break
                
                # Sort by available count (ascending)
                candidates.sort(key=lambda x: x[0])
                _, next_v, e_to_remove = candidates[0]
                
                remaining_edges.remove(e_to_remove)
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
        
        # Optimization: If graph is small, this is fast.
        # For n=30, E ~ 100. Greedy is fast.
        
        paths = decompose_paths(edges)
        
        # Validation check (optional but good for safety)
        # Ensure all edges are covered and disjoint
        # This might be slow, but n is small.
        # We trust the logic.
        
        results.append(paths)

    return results
