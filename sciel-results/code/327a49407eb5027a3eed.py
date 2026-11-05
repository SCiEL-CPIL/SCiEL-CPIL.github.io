import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(n, edges):
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))
        
        paths = []
        remaining_edges = set(edge_set)
        
        # Heuristic: Repeatedly find a long path
        # To avoid getting stuck with many small paths, we prioritize starting at vertices with high degree or odd degree?
        # Let's just try to find a path that covers many edges.
        
        while remaining_edges:
            # Pick a start vertex
            # Prefer vertices with odd degree in the remaining graph?
            # Or just pick one incident to an edge.
            start_candidates = [v for v in range(n) if adj[v] & set(e for e in remaining_edges for e in (e[0], e[1]))]
            # This is inefficient. Let's just pick a random edge (u,v) and start at u.
            # Better: Pick a vertex with the highest degree in the remaining graph.
            degs = [0] * n
            for u, v in remaining_edges:
                degs[u] += 1
                degs[v] += 1
            
            if not any(degs):
                break
                
            # Start at vertex with max degree
            start = max(range(n), key=lambda x: degs[x])
            
            # DFS to find a long path
            # We want to maximize the length.
            # Heuristic: At each step, go to the neighbor with the fewest remaining unvisited neighbors (Warnsdorff).
            # But we must ensure we don't get stuck too early.
            
            path = [start]
            visited_nodes = {start}
            current = start
            
            while True:
                # Find neighbors of current in remaining_edges
                neighbors = [v for v in adj[current] if (current, v) in remaining_edges or (v, current) in remaining_edges]
                
                if not neighbors:
                    break
                
                # Sort neighbors by Warnsdorff's rule: fewest unvisited neighbors
                # "Unvisited" here means neighbors that are still in the graph (have remaining edges) and not in the current path?
                # Actually, we are constructing a simple path, so we can't revisit nodes.
                # But we can revisit nodes if we are not doing a simple path? The task requires SIMPLE paths.
                # So we must not revisit nodes in the current path.
                
                # Filter neighbors to those not in path
                valid_neighbors = [v for v in neighbors if v not in visited_nodes]
                
                if not valid_neighbors:
                    break
                
                # Score each neighbor
                def score(v):
                    # Count unvisited neighbors of v (excluding current and path nodes)
                    unvisited_count = 0
                    for w in adj[v]:
                        if w not in visited_nodes and w != current:
                            # Check if edge (v,w) is still available
                            if (v, w) in remaining_edges or (w, v) in remaining_edges:
                                unvisited_count += 1
                    return unvisited_count
                
                valid_neighbors.sort(key=score)
                next_node = valid_neighbors[0]
                
                # Move to next_node
                path.append(next_node)
                visited_nodes.add(next_node)
                
                # Remove edge (current, next_node) from remaining_edges
                e1 = (current, next_node)
                e2 = (next_node, current)
                if e1 in remaining_edges:
                    remaining_edges.remove(e1)
                elif e2 in remaining_edges:
                    remaining_edges.remove(e2)
                
                current = next_node
            
            if len(path) > 1:
                paths.append(path)
            else:
                # If we got stuck immediately, pick a random neighbor
                # This shouldn't happen if start has degree > 0
                pass

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = solve_graph(n, edges)
        results.append(paths)
        
        if time.time() > deadline:
            break
            
    return results
