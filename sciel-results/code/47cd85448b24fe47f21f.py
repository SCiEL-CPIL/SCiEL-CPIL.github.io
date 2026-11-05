import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(n, edges):
        """
        Decompose graph into minimum number of cycles and single edges.
        Strategy: Use a systematic approach to find edge-disjoint cycles.
        1. Build adjacency.
        2. Repeatedly find a cycle (preferring longer ones) using systematic DFS.
        3. Remove cycle edges.
        4. Remaining edges are single-edge pieces.
        
        To find longer cycles more reliably, we use a multi-start DFS with
        better neighbor ordering (prefer unvisited nodes, then higher degree).
        """
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        
        pieces = []
        used = set()
        
        def remaining_neighbors(v):
            return [x for x in adj[v] if (min(v, x), max(v, x)) not in used]
        
        def has_remaining_edge(v):
            return len(remaining_neighbors(v)) > 0
        
        def find_cycle(start):
            """Find a simple cycle through 'start' using only unused edges.
            Uses DFS with backtracking, preferring to close the cycle when possible
            and exploring neighbors in order of remaining degree (higher first)."""
            path = [start]
            visited = {start}
            best = None
            
            def dfs(node, depth):
                nonlocal best
                if time.time() > deadline:
                    return
                
                # Get remaining neighbors, sorted by remaining degree descending
                nbrs = remaining_neighbors(node)
                # Sort: prefer neighbors that are 'start' (to close cycle) if path long enough,
                # then unvisited, then by remaining degree
                def sort_key(nb):
                    if nb == start and depth >= 3:
                        return (0, 0)  # highest priority: close cycle
                    if nb not in visited:
                        return (1, -len(remaining_neighbors(nb)))
                    return (2, 0)
                nbrs.sort(key=sort_key)
                
                for nb in nbrs:
                    if nb == start and depth >= 3:
                        # Found a cycle
                        candidate = list(path)
                        if best is None or len(candidate) > len(best):
                            best = candidate
                        return  # Stop after first valid cycle for efficiency
                    elif nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        dfs(nb, depth + 1)
                        path.pop()
                        visited.discard(nb)
                    
                    if time.time() > deadline:
                        return
                
                # If we didn't find a cycle in this branch, continue searching
                # But for efficiency, we stop after finding one
            
            dfs(start, 1)
            return best
        
        progress = True
        while progress and time.time() < deadline:
            progress = False
            
            active_vertices = [v for v in range(n) if has_remaining_edge(v)]
            if not active_vertices:
                break
            
            # Try starting from each active vertex, prefer higher remaining degree
            active_vertices.sort(key=lambda v: -len(remaining_neighbors(v)))
            
            for s in active_vertices:
                if time.time() > deadline:
                    break
                if not has_remaining_edge(s):
                    continue
                
                cyc = find_cycle(s)
                if cyc is not None:
                    m = len(cyc)
                    for i in range(m):
                        a, b = cyc[i], cyc[(i + 1) % m]
                        used.add((min(a, b), max(a, b)))
                    pieces.append(cyc)
                    progress = True
        
        # Leftover edges
        for (u, v) in edge_set:
            if (u, v) not in used:
                pieces.append([u, v])
        
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline:
            break
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose_graph(n, edges)
        results.append(pieces)
    
    while len(results) < 3:
        idx = len(results)
        n, p, seed = specs[idx]
        edges = instance(n, p, seed)
        results.append([[list(e)] for e in edges])
    
    return results
