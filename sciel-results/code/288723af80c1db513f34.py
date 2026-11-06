import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140  # 10s safety margin
    
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def longest_path(adj, remaining_edges):
        """Find the longest simple path in the graph defined by remaining_edges."""
        # Build adjacency from remaining edges
        adj = defaultdict(set)
        for u, v in remaining_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_path = []
        n = max(adj.keys()) + 1 if adj else 0
        
        # Try from each vertex as start
        for start in range(n):
            if not adj[start]:
                continue
            stack = [(start, [start], set([start]))]
            while stack:
                node, path, visited = stack.pop()
                if len(path) > len(best_path):
                    best_path = path[:]
                for neighbor in sorted(adj[node], reverse=True):
                    if neighbor not in visited and (min(node, neighbor), max(node, neighbor)) in remaining_edges:
                        new_visited = visited | {neighbor}
                        new_path = path + [neighbor]
                        stack.append((neighbor, new_path, new_visited))
        
        return best_path
    
    def decompose_greedy(edges, max_time=5):
        """Greedy: repeatedly extract longest simple path."""
        remaining = set(edges)
        paths = []
        t0 = time.time()
        
        while remaining and time.time() - t0 < max_time:
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            
            # Find longest simple path
            best_path = None
            n = max(max(e) for e in remaining) + 1 if remaining else 0
            
            # Use DFS with memoization-like pruning
            for start in range(n):
                if not adj.get(start):
                    continue
                # DFS
                visited = set([start])
                path = [start]
                
                def dfs(node):
                    nonlocal best_path
                    if best_path is None or len(path) > len(best_path):
                        if best_path is None or len(path) > len(best_path):
                            best_path = path[:]
                    for neighbor in adj.get(node, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            path.append(neighbor)
                            dfs(neighbor)
                            path.pop()
                            visited.remove(neighbor)
                
                dfs(start)
            
            if not best_path or len(best_path) < 2:
                # Can't find any path, take remaining edges as individual paths
                for u, v in remaining:
                    paths.append([u, v])
                remaining.clear()
                break
            
            # Remove edges from best_path
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                remaining.discard(e)
            paths.append(best_path)
        
        # Handle any leftover edges
        for u, v in remaining:
            paths.append([u, v])
        
        return paths
    
    def try_merge(paths, edges_set):
        """Try to merge two paths into one to reduce count."""
        # ... complex, skip for now
        pass
    
    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for n, p, seed in specs:
        if time.time() > DEADLINE:
            # Fallback: each edge as its own path
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(n, p, seed)
        paths = decompose_greedy(edges, max_time=30)
        results.append(paths)
    
    return results
