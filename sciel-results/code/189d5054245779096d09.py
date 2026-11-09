"""Decompose graphs into few cycles and edges."""
import time
import sys
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_long_cycle(adj, edges_set):
        """Find a reasonably long cycle in the graph. Returns list of vertices or None."""
        if not edges_set:
            return None
        
        best_cycle = None
        best_len = 0
        
        # Try from each vertex with degree >= 2
        for start in list(adj.keys()):
            if len(adj[start]) < 2:
                continue
            
            # DFS to find a long cycle through start
            # Use a greedy heuristic: try to build a long path that returns to start
            visited = {start}
            path = [start]
            
            def dfs(current, depth):
                nonlocal best_cycle, best_len
                if depth > 50:  # cap depth for performance
                    return
                
                # Check if we can return to start
                if start in adj[current] and len(path) >= 3:
                    cycle = path + [start]
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                    return
                
                # Try neighbors in order of degree (prefer high-degree for longer cycles)
                neighbors = sorted(adj[current], key=lambda x: -len(adj[x]))
                for nb in neighbors:
                    if nb == start:
                        continue
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        dfs(nb, depth + 1)
                        path.pop()
                        visited.discard(nb)
                    else:
                        # If nb is in path (not start), we found a cycle
                        if nb in path:
                            idx = path.index(nb)
                            cycle = path[idx:] + [nb]
                            if len(cycle) > best_len:
                                best_len = len(cycle)
                                best_cycle = path[idx:]
                        # Also check if we can close to start through visited
                        # This is handled above when nb == start
                
                # Early termination if we've found a very long cycle
                if best_len > 15:
                    return
            
            dfs(start, 0)
        
        return best_cycle
    
    def decompose(edges):
        """Decompose edges into cycles and single edges, minimizing piece count."""
        remaining = set(edges)
        pieces = []
        
        while remaining:
            # Build adjacency from remaining edges
            adj = defaultdict(set)
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            
            # Try to find a long cycle
            cycle = find_long_cycle(adj, remaining)
            
            if cycle is None:
                # No more cycles; add remaining edges as single pieces
                for u, v in remaining:
                    pieces.append([u, v])
                break
            
            # Remove cycle edges from remaining
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = tuple(sorted((u, v)))
                remaining.discard(edge)
            
            pieces.append(cycle)
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > DEADLINE:
            # Fallback: all edges as single pieces
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
