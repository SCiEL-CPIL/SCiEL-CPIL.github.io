"""Decompose graphs into cycles and edges to minimize piece count."""
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 130
    
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_cycle(adj, n, start, max_nodes=50000):
        count = [0]
        best = []
        
        def dfs(current, path, visited):
            if count[0] > max_nodes:
                return
            count[0] += 1
            
            if len(path) >= 3 and start in adj[current]:
                if len(path) > len(best):
                    best[:] = path[:]
            
            for neighbor in sorted(adj[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        dfs(start, [start], {start})
        return best if len(best) >= 3 else None
    
    def decompose(edges, n):
        pieces = []
        remaining = set(tuple(sorted(e)) for e in edges)
        
        if not remaining:
            return pieces
        
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        while remaining and (time.time() - start_time) < DEADLINE:
            found_cycle = None
            best_len = 0
            
            for start in range(n):
                if len(adj[start]) < 2:
                    continue
                
                cycle = find_cycle(adj, n, start)
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    found_cycle = cycle
                
                if best_len >= n:
                    break
            
            if not found_cycle:
                break
            
            for i in range(len(found_cycle)):
                u, v = found_cycle[i], found_cycle[(i+1) % len(found_cycle)]
                edge = tuple(sorted((u, v)))
                remaining.discard(edge)
                adj[u].discard(v)
                adj[v].discard(u)
            
            pieces.append(found_cycle)
        
        for u, v in sorted(remaining):
            pieces.append([u, v])
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)
    
    return results
