"""Decompose random graphs into few cycles and edges."""

def construct():
    import random
    from collections import defaultdict

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        remaining_edges = set(edges)
        
        while remaining_edges:
            # Find a cycle using a greedy DFS with heuristic
            # Heuristic: prefer neighbors with lower degree to extend the path
            best_cycle = None
            best_len = 0
            
            # Try to start from a vertex with high degree to increase chance of finding a long cycle
            # But the heuristic inside the DFS drives the length.
            # We will try starting from a few promising vertices.
            
            # Get candidates for start
            candidates = [u for u in adj if adj[u]]
            if not candidates:
                break
            
            # Sort candidates by degree descending
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            # Try top 3 candidates to find a good cycle
            for start in candidates[:3]:
                cycle = _find_cycle_heuristic(adj, start)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                    if best_len == len(remaining_edges): # Found a Hamiltonian cycle (unlikely)
                        break
            
            if best_cycle:
                # Remove edges of the cycle
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = tuple(sorted((u, v)))
                    remaining_edges.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
                pieces.append(best_cycle)
            else:
                # No cycle found, take a single edge
                edge = next(iter(remaining_edges))
                remaining_edges.discard(edge)
                u, v = edge
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append(list(edge))
        
        return pieces

    def _find_cycle_heuristic(adj, start):
        """
        Find a cycle starting from 'start' using DFS.
        Heuristic: Explore neighbors with lower degree first to avoid dead ends and create longer cycles.
        Returns the cycle as a list of vertices, or None.
        """
        visited = set()
        path = []
        parent = {}
        
        def dfs(u):
            visited.add(u)
            path.append(u)
            
            # Get neighbors, sorted by degree in current graph (ascending)
            # This mimics Warnsdorff's rule
            neighbors = list(adj[u])
            neighbors.sort(key=lambda x: len(adj[x]))
            
            for v in neighbors:
                if v == parent.get(u):
                    continue
                if v in visited:
                    # Check if v is on the current path
                    if v in path:
                        idx = path.index(v)
                        cycle = path[idx:]
                        if len(cycle) >= 3:
                            return cycle
                else:
                    parent[v] = u
                    result = dfs(v)
                    if result:
                        return result
            path.pop()
            return None
        
        return dfs(start)

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
