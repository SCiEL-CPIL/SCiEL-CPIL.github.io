import sys
sys.setrecursionlimit(10000)
from random import Random

def construct():
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle_heuristic(adj, n, start, rng, iterations=50):
        """
        Find a long simple cycle using Warnsdorff's heuristic with restarts.
        """
        best_cycle = None
        
        for _ in range(iterations):
            path = [start]
            visited = {start}
            last = start
            
            while True:
                neighbors = [v for v in adj[last] if v not in visited]
                if not neighbors:
                    break
                
                # Warnsdorff's rule: pick neighbor with fewest onward moves
                def score(v):
                    return len([u for u in adj[v] if u not in visited])
                
                min_score = min(score(v) for v in neighbors)
                candidates = [v for v in neighbors if score(v) == min_score]
                
                # Add randomness
                if rng.random() < 0.2:
                    next_v = rng.choice(neighbors)
                else:
                    next_v = rng.choice(candidates)
                
                path.append(next_v)
                visited.add(next_v)
                last = next_v
            
            if len(path) >= 3 and start in adj[last]:
                if best_cycle is None or len(path) > len(best_cycle):
                    best_cycle = path[:]
                    if len(best_cycle) == n:
                        break
        
        return best_cycle

    def find_any_cycle(adj, n, start):
        """
        Find any cycle in the component containing start using BFS.
        Returns list of vertices in the cycle, or None.
        """
        if not adj[start]:
            return None
            
        visited = {start: -1}
        parent = {start: -1}
        queue = [start]
        
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in visited:
                    visited[v] = u
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Back edge found
                    # Cycle involves u and v
                    # Trace back from u to v
                    cycle = [u, v]
                    curr = u
                    while curr != v:
                        curr = parent[curr]
                        cycle.append(curr)
                    # The cycle is v -> ... -> u -> v
                    # The vertices are v, ..., u
                    # cycle list currently: [u, v, p(v), ..., p(u)]? No.
                    # Let's reconstruct carefully.
                    # parent chain from u: u, p(u), p(p(u)), ... until v.
                    # So the path is u -> p(u) -> ... -> v.
                    # The cycle is v -> ... -> u -> v.
                    # Vertices: v, ..., u.
                    # Let's just return the list of vertices in the order they appear in the cycle.
                    # Order: v, ..., u.
                    # My loop: curr=u, append u? No, I initialized cycle=[u,v].
                    # Let's do it this way:
                    path = [u]
                    curr = u
                    while curr != v:
                        curr = parent[curr]
                        path.append(curr)
                    # path is [u, ..., v]
                    # The cycle is path (u...v) plus edge v-u.
                    # So the cycle vertices are path.
                    return path
        return None

    def decompose(edges):
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        adj = {i: set() for i in range(n)}
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
        
        pieces = []
        rng = Random(42)
        
        while edge_set:
            active_vertices = [v for v in range(n) if adj[v]]
            if not active_vertices:
                break
                
            best_cycle = None
            best_len = 0
            
            # Prioritize high degree vertices
            active_vertices.sort(key=lambda x: len(adj[x]), reverse=True)
            
            for start in active_vertices[:10]:
                cycle = find_cycle_heuristic(adj, n, start, rng, iterations=30)
                if cycle:
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                        if best_len == n:
                            break
            
            if best_cycle is None:
                # Fallback
                for start in active_vertices:
                    cycle = find_any_cycle(adj, n, start)
                    if cycle:
                        best_cycle = cycle
                        break
            
            if best_cycle is None:
                for u, v in sorted(edge_set):
                    pieces.append([u, v])
                break
            
            # Remove cycle edges
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                edge = (min(u, v), max(u, v))
                if edge in edge_set:
                    edge_set.remove(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
            
            pieces.append(best_cycle[:])
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
