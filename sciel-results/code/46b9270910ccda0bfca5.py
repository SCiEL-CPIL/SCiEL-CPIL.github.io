import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(edges_set, n, max_attempts=50):
        """
        Attempts to find a long cycle in the graph defined by edges_set.
        Returns a list of vertices representing the cycle, or None.
        """
        if not edges_set:
            return None
            
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find a connected component with at least 3 edges
        visited = set()
        component_nodes = []
        
        # Pick a random start node
        start_node = random.choice(list(edges_set)[0])
        
        # BFS/DFS to get component
        stack = [start_node]
        comp_visited = set()
        while stack:
            node = stack.pop()
            if node in comp_visited:
                continue
            comp_visited.add(node)
            component_nodes.append(node)
            for neighbor in adj[node]:
                if neighbor not in comp_visited:
                    stack.append(neighbor)
        
        # Check if this component actually has a cycle
        # A component with V vertices and E edges has a cycle if E >= V
        # We can just try to find one.
        
        best_cycle = None
        best_len = 0
        
        for _ in range(max_attempts):
            # Randomized DFS
            # Start from a random node in the component
            start = random.choice(component_nodes)
            path = [start]
            visited_path = {start}
            
            def dfs(node, path, visited_path):
                nonlocal best_cycle, best_len
                for neighbor in adj[node]:
                    if neighbor in visited_path:
                        # Cycle found
                        # Extract cycle: from neighbor to end of path
                        idx = path.index(neighbor)
                        cycle = path[idx:]
                        if len(cycle) > best_len:
                            best_len = len(cycle)
                            best_cycle = cycle[:]
                        # Don't return immediately, try to find longer
                    else:
                        visited_path.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited_path)
                        path.pop()
                        visited_path.remove(neighbor)
            
            dfs(start, path, visited_path)
            
            if time.time() > deadline:
                break

        return best_cycle

    results = []
    
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Fallback to all edges if we run out of time
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        n = spec[0]
        remaining_edges = set(tuple(e) for e in edges)
        pieces = []
        
        while remaining_edges:
            if time.time() > deadline:
                # Add all remaining as single edges
                for e in remaining_edges:
                    pieces.append(list(e))
                break
                
            # Find a cycle in the remaining graph
            cycle = find_longest_cycle(remaining_edges, n, max_attempts=100)
            
            if cycle:
                # Convert cycle vertices to edges
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = tuple(sorted((u, v)))
                    if edge in remaining_edges:
                        cycle_edges.add(edge)
                
                if cycle_edges:
                    pieces.append(cycle)
                    remaining_edges -= cycle_edges
                else:
                    # Cycle edges not in remaining? Should not happen.
                    # Fallback: pick one edge
                    e = next(iter(remaining_edges))
                    pieces.append(list(e))
                    remaining_edges.remove(e)
            else:
                # No cycle found, pick a single edge
                e = next(iter(remaining_edges))
                pieces.append(list(e))
                remaining_edges.remove(e)
                
        results.append(pieces)
        
    return results
