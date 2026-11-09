import time
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose(edges):
        if not edges:
            return []
        
        adj = defaultdict(list)
        edge_set = set()
        for e in edges:
            u, v = e
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add(tuple(sorted(e)))
        
        # Track remaining edges
        remaining = set(edge_set)
        paths = []
        
        # Compute degrees
        deg = defaultdict(int)
        for e in remaining:
            deg[e[0]] += 1
            deg[e[1]] += 1
        
        def find_path(start, visited_edges):
            """Find a path starting from start using DFS, avoiding visited edges."""
            path = [start]
            visited = {start}
            
            def dfs(node):
                if time.time() > deadline:
                    return False
                for nbr in adj[node]:
                    e = tuple(sorted((node, nbr)))
                    if e not in visited_edges and e in remaining:
                        if nbr not in visited:
                            visited.add(nbr)
                            path.append(nbr)
                            if dfs(nbr):
                                return True
                            path.pop()
                            visited.discard(nbr)
                        else:
                            # Can't revisit, but this edge might still be usable
                            # Actually for simple paths, we can't revisit vertices
                            pass
                return False
            
            dfs(start)
            return path if len(path) > 1 else None
        
        def find_longest_path(remaining_edges, adj_map):
            """Find the longest path in the subgraph of remaining edges."""
            if not remaining_edges:
                return None
            
            # Build subgraph adjacency
            sub_adj = defaultdict(set)
            for e in remaining_edges:
                u, v = e
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            # Try from each vertex, prefer odd-degree
            vertices = list(sub_adj.keys())
            # Sort: odd degree first, then by degree descending
            vertices.sort(key=lambda v: (- (deg.get(v, 0) % 2 == 0), -deg.get(v, 0)))
            
            best_path = None
            best_len = 0
            
            for start in vertices:
                if not sub_adj[start]:
                    continue
                # DFS to find longest path
                path = [start]
                visited = {start}
                
                def dfs(node):
                    nonlocal best_path, best_len
                    if time.time() > deadline:
                        return
                    if len(path) > best_len:
                        best_len = len(path)
                        best_path = list(path)
                    for nbr in sub_adj[node]:
                        if nbr not in visited:
                            visited.add(nbr)
                            path.append(nbr)
                            dfs(nbr)
                            path.pop()
                            visited.discard(nbr)
                
                dfs(start)
                if time.time() > deadline:
                    break
            
            return best_path
        
        # Greedy: repeatedly find longest path
        while remaining:
            if time.time() > deadline:
                break
            
            path = find_longest_path(remaining, adj)
            if path is None or len(path) < 2:
                # Single edges left
                for e in list(remaining):
                    paths.append([e[0], e[1]])
                    remaining.discard(e)
                break
            
            # Remove edges used in path
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                remaining.discard(e)
            
            paths.append(path)
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: each edge as its own path
            edges = instance(*spec)
            results.append([[e[0], e[1]] for e in edges])
            continue
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    
    return results
