import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_path_dfs(start, end, remaining_edges, adj, max_depth=50):
        """Find a simple path from start to end using only edges in remaining_edges."""
        if start == end:
            return [start]
        
        visited = {start}
        path = [start]
        
        def dfs(node, path, visited):
            if node == end:
                return list(path)
            if len(path) > max_depth:
                return None
            
            neighbors = list(adj[node])
            # Sort neighbors: prefer those with fewer options (dead-end heuristic)
            def neighbor_score(nb):
                count = 0
                for nb2 in adj[nb]:
                    e = tuple(sorted((nb, nb2)))
                    if e in remaining_edges and nb2 not in visited:
                        count += 1
                return count
            
            neighbors.sort(key=neighbor_score)
            
            for nb in neighbors:
                if nb in visited:
                    continue
                e = tuple(sorted((node, nb)))
                if e not in remaining_edges:
                    continue
                
                # Try this edge
                remaining_edges.discard(e)
                visited.add(nb)
                path.append(nb)
                
                result = dfs(nb, path, visited)
                if result is not None:
                    return result
                
                # Backtrack
                path.pop()
                visited.discard(nb)
                remaining_edges.add(e)
            
            return None
        
        return dfs(start, path, visited)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        paths = []
        
        # Compute degrees
        degree = defaultdict(int)
        for u, v in remaining_edges:
            degree[u] += 1
            degree[v] += 1
        
        # Find odd-degree vertices
        odd_vertices = [v for v in degree if degree[v] % 2 == 1]
        
        # Pair up odd vertices
        odd_vertices.sort()
        pairs = []
        for i in range(0, len(odd_vertices) - 1, 2):
            pairs.append((odd_vertices[i], odd_vertices[i + 1]))
        
        # For each pair, try to find a path
        for (a, b) in pairs:
            if time.time() - start_time > time_limit - 5:
                break
            # Make a copy to try path finding
            # We'll use the actual remaining_edges with backtracking
            path = find_path_dfs(a, b, remaining_edges, adj, max_depth=40)
            if path and len(path) >= 2:
                paths.append(path)
                # Update degrees
                for i in range(len(path) - 1):
                    degree[path[i]] -= 1
                    degree[path[i + 1]] -= 1
                # Re-identify odd vertices for next round? No, we already paired.
            else:
                # Can't find path between a and b; will handle with greedy later
                pass
        
        # Greedy: cover remaining edges
        while remaining_edges:
            if time.time() - start_time > time_limit - 2:
                # Emergency: add remaining edges as individual paths
                for e in remaining_edges:
                    paths.append(list(e))
                break
            
            # Pick an edge to start a path
            # Choose edge where one endpoint has highest remaining degree
            def edge_key(e):
                u, v = e
                du = sum(1 for nb in adj[u] if tuple(sorted((u, nb))) in remaining_edges)
                dv = sum(1 for nb in adj[v] if tuple(sorted((v, nb))) in remaining_edges)
                return -(du + dv)
            
            # Sort edges and pick the best
            best_edge = None
            best_key = float('inf')
            for e in remaining_edges:
                k = edge_key(e)
                if k < best_key:
                    best_key = k
                    best_edge = e
            
            if best_edge is None:
                break
            
            u, v = best_edge
            # Extend path from both ends
            path = [u, v]
            remaining_edges.discard(best_edge)
            
            # Extend from u end
            while True:
                current = path[0]
                candidates = [nb for nb in adj[current] 
                             if tuple(sorted((current, nb))) in remaining_edges 
                             and nb not in set(path)]
                if not candidates:
                    break
                # Pick neighbor with highest remaining degree
                candidates.sort(key=lambda nb: -sum(1 for nb2 in adj[nb] 
                                                    if tuple(sorted((nb, nb2))) in remaining_edges))
                next_node = candidates[0]
                path.insert(0, next_node)
                remaining_edges.discard(tuple(sorted((current, next_node))))
            
            # Extend from v end
            while True:
                current = path[-1]
                candidates = [nb for nb in adj[current] 
                             if tuple(sorted((current, nb))) in remaining_edges 
                             and nb not in set(path)]
                if not candidates:
                    break
                candidates.sort(key=lambda nb: -sum(1 for nb2 in adj[nb] 
                                                    if tuple(sorted((nb, nb2))) in remaining_edges))
                next_node = candidates[0]
                path.append(next_node)
                remaining_edges.discard(tuple(sorted((current, next_node))))
            
            paths.append(path)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
    return results
