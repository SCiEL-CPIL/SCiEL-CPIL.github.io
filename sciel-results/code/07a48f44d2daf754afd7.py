import time
from collections import defaultdict, deque
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def build_adj(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_longest_path_greedy(edges, n, num_trials=50):
        """
        Find a long simple path in the graph defined by edges.
        Uses randomized greedy DFS.
        """
        if not edges:
            return []
        
        adj = build_adj(edges)
        best_path = []
        
        for _ in range(num_trials):
            # Start from a random vertex
            start = random.choice(list(range(n)))
            path = [start]
            visited = {start}
            current = start
            
            while True:
                # Get unvisited neighbors
                neighbors = [v for v in adj.get(current, []) if v not in visited]
                if not neighbors:
                    break
                
                # Heuristic: pick neighbor with highest degree in remaining graph
                # Or just random? Random with restarts is often better for finding long paths
                # Let's try picking the one with highest degree to keep options open
                # Actually, for longest path, picking the one with *lowest* degree might be better to avoid getting stuck?
                # No, picking the one with highest degree usually leads to longer paths in random graphs.
                # Let's try a mix: 50% highest degree, 50% random.
                
                # Calculate degrees in the induced subgraph of unvisited nodes?
                # That's expensive. Let's just use current adjacency degree.
                max_deg = -1
                best_neighbor = None
                for v in neighbors:
                    deg = len(adj.get(v, []))
                    if deg > max_deg:
                        max_deg = deg
                        best_neighbor = v
                
                if random.random() < 0.5 and best_neighbor is not None:
                    next_node = best_neighbor
                else:
                    next_node = random.choice(neighbors)
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            if len(path) > len(best_path):
                best_path = path
                
        return best_path

    def decompose_graph(edges):
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        remaining = set(edges)
        paths = []
        
        while remaining:
            # Check time
            if time.time() > DEADLINE - 5:
                # Fallback: single edges
                paths.extend([[u, v] for u, v in remaining])
                break
                
            path = find_longest_path_greedy(remaining, n, num_trials=20)
            
            if not path:
                # Should not happen if remaining is not empty
                # Fallback: take any edge
                u, v = next(iter(remaining))
                path = [u, v]
            
            # Remove edges of this path from remaining
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = (min(u, v), max(u, v))
                if e in remaining:
                    remaining.discard(e)
            
            if len(path) >= 2:
                paths.append(path)
                
        return paths

    def verify_decomposition(edges, paths):
        used = set()
        for path in paths:
            if len(set(path)) != len(path):
                return False
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                e = (min(u, v), max(u, v))
                if e in used:
                    return False
                if e not in edges:
                    return False
                used.add(e)
        return used == set(edges)

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for n, p, seed in specs:
        if time.time() > DEADLINE - 15:
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(n, p, seed)
        
        try:
            paths = decompose_graph(edges)
            
            if not verify_decomposition(set(edges), paths):
                # Fallback
                paths = [[u, v] for u, v in edges]
            
            results.append(paths)
        except Exception:
            paths = [[u, v] for u, v in edges]
            results.append(paths)
    
    return results
