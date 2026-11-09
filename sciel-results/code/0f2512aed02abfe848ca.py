import random
import sys

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def solve_graph(n, edges):
        # Build adjacency list and degree count
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            # Find vertex with minimum degree > 0
            min_deg = float('inf')
            start_node = None
            for u in range(n):
                deg = len(adj[u])
                if deg > 0 and deg < min_deg:
                    min_deg = deg
                    start_node = u
            
            if start_node is None:
                break
                
            path = [start_node]
            current = start_node
            visited_in_path = {start_node}
            
            while True:
                # Find neighbor with max degree
                best_next = None
                max_deg = -1
                for v in adj[current]:
                    if v not in visited_in_path:
                        if len(adj[v]) > max_deg:
                            max_deg = len(adj[v])
                            best_next = v
                
                if best_next is None:
                    break
                
                # Add edge to path
                path.append(best_next)
                visited_in_path.add(best_next)
                
                # Remove edge from remaining and adj
                u = current
                v = best_next
                edge_key = tuple(sorted((u, v)))
                remaining_edges.remove(edge_key)
                
                adj[u].remove(v)
                adj[v].remove(u)
                
                current = best_next
                
            paths.append(path)
            
        return paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = solve_graph(n, edges)
        results.append(paths)
        
    return results
