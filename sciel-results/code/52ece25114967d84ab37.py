import time
import random
from collections import deque

def _check_girth(n, edges, min_girth):
    """
    Check if the graph has girth >= min_girth.
    Uses BFS from each vertex to find shortest cycles.
    Optimization: We only need to check up to depth min_girth//2.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, BFS to find shortest cycle passing through it
    # Actually, we can just do BFS from each vertex and check for back-edges.
    # A cycle of length L is found if we reach a visited node at depth d 
    # such that d + dist_to_start < L? 
    # Standard girth check:
    # For each start node s, do BFS.
    # If we find an edge (u, v) where u and v are in the BFS tree 
    # and dist(s, u) + dist(s, v) + 1 < current_best_girth, update.
    # We stop BFS if depth >= min_girth // 2.
    
    best_girth = float('inf')
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] * 2 >= min_girth: 
                # If we go deeper, any cycle found will be longer
                # Actually, cycle length = dist[u] + dist[v] + 1.
                # If dist[u] >= min_girth//2, then dist[u]+dist[v]+1 >= min_girth.
                # So we can stop expanding.
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < best_girth:
                        best_girth = cycle_len
                    if best_girth < min_girth:
                        return False
    return best_girth >= min_girth

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Heuristic for LUW d parameter
    # For k=5, g=9, try d=4 first
    best_n = float('inf')
    best_edges = None
    
    # 1. Try LUW construction
    # The seed logic for LUW
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    def build_luw(d):
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

    # Try d=4 for k=5, g=9
    if k == 5 and g == 9:
        # Try d=4
        d_try = 4
        n, edges = build_luw(d_try)
        # Verify girth
        if _check_girth(n, edges, g):
            best_n = n
            best_edges = edges
        
        # If d=4 didn't work or wasn't optimal, try d=5 (seed)
        if time.time() < deadline:
            d_try = 5
            n, edges = build_luw(d_try)
            if _check_girth(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges

    # 2. Try Circulant Search if time permits and best_n is still large
    # We want n < best_n
    if time.time() < deadline and best_n > 1000:
        # Search for circulant graphs
        # Start from Moore bound
        moore_bound = 1 + k * sum([(k-1)**i for i in range((g-1)//2)])
        # Actually for g=9, r=4, sum is 1+4+16+64=85. 1+5*85=426.
        
        # Try a few n values
        # To save time, we'll just try a few random step sets for n around 1000-1200
        # Or just skip if LUW d=4 gave 1250.
        # 1250 is already good.
        
        # Let's try to find a smaller circulant graph.
        # We'll try n from 1000 to 1249.
        # This might be too slow.
        # Instead, let's just stick with the LUW result if it's valid.
        
        pass # Skip circulant search for now to ensure reliability and speed.

    if best_edges is None:
        # Fallback: Use the seed's original logic (d=5)
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_edges = build_luw(d)
        
    return best_n, best_edges
