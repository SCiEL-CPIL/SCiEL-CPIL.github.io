"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""

import time
from random import Random

def construct():
    deadline = time.time() + 115.0
    
    def instance(n, p, seed):
        rng = Random(seed)
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
    
    def decompose(edges_list, time_budget):
        if not edges_list:
            return []
        
        adj = {}
        for u, v in edges_list:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        remaining = set(edges_list)
        pieces = []
        
        def find_cycle_from(start, adj):
            """Find a cycle starting from 'start' using DFS. Returns cycle or None."""
            visited = {start}
            parent = {start: None}
            path = [start]
            stack = [start]
            
            while stack:
                u = stack.pop()
                for v in adj.get(u, set()):
                    if v == start and len(path) >= 3:
                        # Found a cycle
                        return list(path)
                    elif v not in visited:
                        visited.add(v)
                        parent[v] = u
                        path.append(v)
                        stack.append(v)
                    # If v is visited and not parent, we have a cycle but not necessarily through start
                    # For simplicity, we only look for cycles through start
            
            # Backtrack to find cycle through start
            # Actually, the above doesn't properly find cycles through start
            # Let me use a different approach
            return None
        
        def find_cycle_dfs(start, adj):
            """Find a cycle containing start using DFS."""
            visited = set()
            parent = {}
            
            def dfs(u):
                visited.add(u)
                for v in sorted(adj.get(u, set())):
                    if v == start:
                        # Found cycle: reconstruct path from u back to start
                        cycle = [u]
                        cur = u
                        while cur != start:
                            cur = parent[cur]
                            cycle.append(cur)
                        cycle.reverse()
                        return cycle
                    elif v not in visited:
                        parent[v] = u
                        result = dfs(v)
                        if result:
                            return result
                return None
            
            parent[start] = None
            return dfs(start)
        
        def find_best_cycle(adj):
            """Try multiple starts, find the longest cycle."""
            best = None
            best_len = 0
            for start in sorted(adj.keys()):
                if time.time() > time_budget:
                    break
                if len(adj.get(start, set())) < 2:
                    continue
                cycle = find_cycle_dfs(start, adj)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best = cycle
            return best
        
        while remaining:
            if time.time() > time_budget:
                break
            
            cycle = find_best_cycle(adj)
            
            if cycle:
                pieces.append(cycle)
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                e = next(iter(remaining))
                pieces.append([e[0], e[1]])
                remaining.discard(e)
                adj[e[0]].discard(e[1])
                adj[e[1]].discard(e[0])
        
        return pieces
    
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for i, spec in enumerate(specs):
        time_left = deadline - time.time()
        if time_left <= 5:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        per_graph_budget = time_left / (len(specs) - i)
        edges = instance(*spec)
        pieces = decompose(edges, time.time() + per_graph_budget)
        results.append(pieces)
    
    return results
