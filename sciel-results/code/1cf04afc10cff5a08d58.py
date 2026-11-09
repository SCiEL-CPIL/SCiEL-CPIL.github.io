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

    def find_simple_path(start, end, adj, remaining_edges):
        """Find a simple path from start to end using only edges in remaining_edges."""
        if start == end:
            return [start]
        
        visited = {start}
        path = [start]
        
        def dfs(node):
            if node == end:
                return True
            
            neighbors = list(adj[node])
            # Heuristic: try neighbors with fewer options first?
            # Or just standard DFS
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
                
                if dfs(nb):
                    return True
                
                # Backtrack
                path.pop()
                visited.discard(nb)
                remaining_edges.add(e)
            
            return False
        
        if dfs(start):
            return path
        return None

    def find_simple_cycle(adj, remaining_edges):
        """Find any simple cycle in the graph formed by remaining_edges."""
        # Pick a vertex with degree > 0
        start_node = None
        for u, v in remaining_edges:
            start_node = u
            break
        if start_node is None:
            return None

        visited = {start_node}
        path = [start_node]
        
        def dfs(node):
            for nb in adj[node]:
                e = tuple(sorted((node, nb)))
                if e not in remaining_edges:
                    continue
                
                if nb == start_node and len(path) > 1:
                    # Found a cycle
                    return list(path) + [start_node]
                
                if nb in visited:
                    continue
                
                visited.add(nb)
                path.append(nb)
                
                result = dfs(nb)
                if result:
                    return result
                
                path.pop()
                visited.discard(nb)
            
            return None
        
        return dfs(start_node)

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
            path = find_simple_path(a, b, adj, remaining_edges)
            if path:
                paths.append(path)
            else:
                # This should not happen if graph is connected and a,b are in same component
                # But just in case, we handle it in the cycle/greedy phase
                pass
        
        # Decompose remaining edges (which should have all even degrees) into cycles
        while remaining_edges:
            if time.time() - start_time > time_limit - 2:
                # Emergency: add remaining edges as individual paths (might be invalid if not simple paths, but better than crash)
                # Actually, if we just add edges, they are paths of length 1.
                for e in remaining_edges:
                    paths.append(list(e))
                break
            
            cycle = find_simple_cycle(adj, remaining_edges)
            if not cycle:
                # Should not happen if graph is Eulerian, but if it does, fallback to greedy edge picking
                # Pick any edge
                u, v = next(iter(remaining_edges))
                paths.append([u, v])
                remaining_edges.discard(tuple(sorted((u, v))))
                continue
            
            # Remove cycle edges from remaining_edges
            for i in range(len(cycle) - 1):
                e = tuple(sorted((cycle[i], cycle[i+1])))
                remaining_edges.discard(e)
            
            # Convert cycle to path
            # Cycle is [v0, v1, ..., vk, v0]
            # Path is [v0, v1, ..., vk]
            path_from_cycle = cycle[:-1]
            paths.append(path_from_cycle)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
    return results
