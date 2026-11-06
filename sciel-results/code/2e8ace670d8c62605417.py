import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Track remaining edges
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: return remaining edges as individual paths
                # This ensures validity, though suboptimal
                for e in remaining_edges:
                    paths.append(list(e))
                return paths

            # Find a starting vertex
            # Prefer vertices with low degree in the remaining graph
            candidates = [v for v in adj if any(tuple(sorted((v, nb))) in remaining_edges for nb in adj[v])]
            if not candidates:
                break
            
            # Sort candidates by degree in remaining graph
            def get_deg(v):
                count = 0
                for nb in adj[v]:
                    if tuple(sorted((v, nb))) in remaining_edges:
                        count += 1
                return count
            
            candidates.sort(key=get_deg)
            start = candidates[0]
            
            # Greedy extension
            path = [start]
            current = start
            visited_in_path = {start}
            
            while True:
                # Find next vertex
                next_v = None
                min_deg = float('inf')
                
                # Get neighbors with remaining edges
                neighbors = [nb for nb in adj[current] if tuple(sorted((current, nb))) in remaining_edges]
                
                if not neighbors:
                    break
                
                # Choose neighbor with minimum remaining degree
                # Tie-breaker: random or arbitrary
                neighbors.sort(key=lambda nb: get_deg(nb))
                
                # If there are multiple with same degree, pick one
                # To improve, we could look ahead, but let's keep it simple first
                next_v = neighbors[0]
                
                # Add edge to path
                edge = tuple(sorted((current, next_v)))
                remaining_edges.remove(edge)
                
                path.append(next_v)
                visited_in_path.add(next_v)
                current = next_v
                
                # If we revisit a vertex, it's not a simple path?
                # Wait, the problem says "Paths must be simple".
                # My greedy above allows revisiting vertices if I'm not careful.
                # `visited_in_path` check is needed.
                
                # RESTART: The simple greedy above might get stuck in a cycle or revisit a vertex.
                # I need to ensure the path is simple (no repeated vertices).
                
                pass # Placeholder, logic below

            # The logic above was flawed for simple paths.
            # Let's rewrite the decomposition loop properly.
            
            return paths # Placeholder

    # Proper Decomposition
    def decompose_graph_v2(edges):
        if not edges:
            return []
        
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    paths.append(list(e))
                return paths

            # Find start vertex: min degree in remaining graph
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
                
            if not active_vertices:
                break

            # Calculate degrees
            degs = {v: 0 for v in active_vertices}
            for u, v in remaining_edges:
                degs[u] += 1
                degs[v] += 1
                
            start = min(active_vertices, key=lambda x: degs[x])
            
            # Greedy Simple Path Extension
            path = [start]
            current = start
            visited = {start}
            
            while True:
                # Find available neighbors
                avail = [nb for nb in adj[current] 
                         if tuple(sorted((current, nb))) in remaining_edges 
                         and nb not in visited]
                
                if not avail:
                    break
                
                # Choose next vertex
                # Heuristic: Pick neighbor with lowest degree to keep graph "open"
                # Or highest?
                # Let's try lowest degree (leaf-preference)
                avail.sort(key=lambda x: degs[x])
                next_v = avail[0]
                
                # Update state
                edge = tuple(sorted((current, next_v)))
                remaining_edges.remove(edge)
                
                # Update degrees for subsequent steps
                degs[current] -= 1
                degs[next_v] -= 1
                
                path.append(next_v)
                visited.add(next_v)
                current = next_v
                
            paths.append(path)
            
        return paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for spec in specs:
        edges = instance(*spec)
        paths = decompose_graph_v2(edges)
        results.append(paths)
        
    return results
