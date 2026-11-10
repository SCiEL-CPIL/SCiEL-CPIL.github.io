import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    # Strategy 1: Use known explicit (4,10)-cage if parameters match
    if k == 4 and g == 10:
        n_cage = 320
        # Generators for the (4,10)-cage circulant graph
        # Note: These generators are a candidate. We must verify.
        generators = [1, 12, 15, 25]
        
        # Construct edges
        edge_set = set()
        for v in range(n_cage):
            for s in generators:
                u = (v + s) % n_cage
                if v < u:
                    edge_set.add((v, u))
                else:
                    edge_set.add((u, v))
        
        edges_cage = list(edge_set)
        
        # Verify girth is at least g
        if _verify_girth(n_cage, edges_cage, g):
            return n_cage, edges_cage
        # If verification fails, fall back to LUW

    # Strategy 2: Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    q = k
    # For girth g, we need d such that the graph has girth >= g.
    # LUW graph CD(d, q) has girth 2d or 2d+1.
    # For g=10, we need d=5 (girth 10) or d=4 (girth 8, too small).
    # So d=5.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Precompute field operations for speed
    if q == 4:
        # GF(4) represented as 0,1,2,3 with XOR addition and table multiplication
        add_tbl = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        mul_tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def add(x, y): return add_tbl[x][y]
        def sub(x, y): return add_tbl[x][y] # In char 2, sub = add
        def mul(x, y): return mul_tbl[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # BFS to build graph
    i = 0
    while i < len(order):
        if time.time() > deadline:
            break
        vertex = order[i]
        i += 1
        v_side, v_x = vertex
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if v_side == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _verify_girth(n, edges, g):
    """
    Verify that the graph with n vertices and given edges has girth at least g.
    Uses BFS from each vertex to detect cycles of length < g.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check for cycles of length < g
    # We perform BFS up to depth floor((g-1)/2)
    max_depth = (g - 1) // 2
    
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle
                    # The cycle length is dist[u] + dist[v] + 1
                    if dist[u] + dist[v] + 1 < g:
                        return False
                        
    return True
