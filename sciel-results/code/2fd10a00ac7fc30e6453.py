import time
import math
import random

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    For k=4, g=10, it searches for a small random 4-regular graph.
    Falls back to Lazebnik-Ustimenko-Woldar construction if search fails or takes too long.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin
    
    if k != 4 or g != 10:
        # Fallback to generic LUW for other parameters if needed, 
        # but the task specifies this task's degree and girth.
        return _luw_construct(k, g)

    # Target size 320 (known smallest cage)
    target_n = 320
    
    # Try to find a random 4-regular graph of size 320 with girth >= 10
    # We use the configuration model to generate candidates.
    
    def _gen_random_4reg(n):
        """Generates a random 4-regular graph on n vertices. Returns edges or None if failed (loops/multiedges)."""
        stubs = []
        for i in range(n):
            for _ in range(4):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        edges = set()
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                return None
            edges.add((u, v))
            
        return list(edges)

    def _has_girth_at_least(n, edges, g):
        """Checks if graph has girth at least g using BFS up to depth (g-2)//2."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For each vertex, do BFS up to depth 4.
        # If we find a cycle of length < g, return False.
        # A cycle of length L < g implies there is an edge (u,v) such that
        # dist_s(u) + dist_s(v) + 1 < g for some s?
        # No, the standard check is:
        # For each vertex s, do BFS.
        # If we find an edge (u,v) such that the path from s to u and s to v
        # are disjoint except at s? No.
        
        # Simpler: For each vertex s, do BFS up to depth 4.
        # If we reach a vertex v at depth d, and v is already visited at depth d' 
        # such that d + d' + 1 < g, we have a cycle.
        # Actually, if we just check if the BFS tree has any "chords" that create short cycles.
        
        # Most efficient for small g:
        # For each vertex s:
        #   dist = [-1]*n
        #   dist[s] = 0
        #   q = [s]
        #   qi = 0
        #   while qi < len(q):
        #     u = q[qi]; qi += 1
        #     du = dist[u]
        #     if du >= 4: continue
        #     for v in adj[u]:
        #       if dist[v] == -1:
        #         dist[v] = du + 1
        #         q.append(v)
        #       else:
        #         if du + dist[v] + 1 < g:
        #           return False
        #   # Also need to check edges between nodes at depth 4?
        #   # If u at 4, v at 4, edge (u,v). Cycle length 4+1+4=9.
        #   # The check above happens when we process u.
        #   # If v is visited, we check.
        #   # If v is not visited, we set dist[v]=5.
        #   # We don't check edge (u,v) for cycle if v is new?
        #   # But if v is new, dist[v] becomes 5.
        #   # Later, if we process v (depth 5), we stop.
        #   # So we miss the edge (u,v) if v was unvisited when u was processed?
        #   # No, if v was unvisited, we set dist[v]=5.
        #   # The edge (u,v) is a tree edge. It doesn't form a cycle.
        #   # The cycle is formed by non-tree edges.
        #   # So checking only when dist[v] != -1 is correct.
        #   # But we need to ensure we catch the case where u and v are both at depth 4.
        #   # When we process u (depth 4), if v is already visited (depth 4), we check.
        #   # If v is not visited, we set dist[v]=5.
        #   # But wait, if v is at depth 4, it must have been visited earlier.
        #   # So the check works.
        
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = [s]
            qi = 0
            while qi < len(q):
                u = q[qi]
                qi += 1
                du = dist[u]
                if du >= 4:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        if du + 1 <= 4:
                            q.append(v)
                    else:
                        if du + dist[v] + 1 < g:
                            return False
        return True

    # Search loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        edges = _gen_random_4reg(target_n)
        if edges is None:
            continue
        
        if _has_girth_at_least(target_n, edges, g):
            return target_n, edges
            
    # Fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original LUW code
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
