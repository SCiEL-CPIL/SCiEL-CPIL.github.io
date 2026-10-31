import time
import random
import itertools
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if k != 3 or g != 16:
        return _seed_construct(k, g)
    
    # Try LUW construction with various d values
    # LUW graph CD(d, q) has 2*q^d vertices and is q-regular
    # Girth is approximately d+5 for odd d, d+6 for even d (need to verify)
    # For q=3, g=16: try d values and check actual girth
    
    best_result = None
    best_n = float('inf')
    
    # Try d from 6 to 12 (girth increases with d)
    for d in range(6, 13):
        if time.time() > deadline - 5:
            break
        n, edges = _luw_construct(3, d)
        if n < best_n:
            # Verify girth
            girth = _verify_girth(n, edges, g)
            if girth >= g:
                best_n = n
                best_result = (n, edges)
                break  # Found a valid one with smaller n, stop
    
    if best_result:
        return best_result
    
    # Fallback: try a larger d that's guaranteed to work
    d = 12
    n, edges = _luw_construct(3, d)
    girth = _verify_girth(n, edges, g)
    if girth >= g:
        return n, edges
    
    # Ultimate fallback: seed
    return _seed_construct(k, g)

def _luw_construct(q, d):
    """Lazebnik-Ustimenko-Woldar construction: 2*q^d vertices, q-regular"""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        mod = q
        add = lambda x, y: (x + y) % mod
        sub = lambda x, y: (x - y) % mod
        mul = lambda x, y: (x * y) % mod
    
    # Build the rule table
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    # Generate all vertices: (side, x_1, ..., x_d) where side in {0,1}
    # side=0: x_1 is free, x_2...x_d determined
    # Actually, LUW: vertices are (side, x) where x in Z_q^d
    # Neighbors of (0, x): (1, y) where y_1 = x_1, and for c>=2: y_c = x_c + y_{rule[c][0]} * x_{rule[c][1]}
    # Neighbors of (1, x): (0, y) where y_1 = x_1, and for c>=2: y_c = x_c - x_{rule[c][0]} * y_{rule[c][1]}
    
    total = 2 * (q ** d)
    
    # Index vertices
    index = {}
    order = []
    
    # Enumerate all vertices
    for side in range(2):
        for coords in itertools.product(range(q), repeat=d):
            v = (side, coords)
            index[v] = len(order)
            order.append(v)
    
    edges = []
    
    for vi, vertex in enumerate(order):
        side, x = vertex
        for first in range(q):
            # Build y
            y = [0] * (d + 1)  # 1-indexed: y[1]...y[d]
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c-1], mul(y[a-1], x[b-1]))
                else:
                    y[c] = sub(x[c-1], mul(x[a-1], y[b-1]))
            other = (1 - side, tuple(y))
            if other in index:
                if side == 0:  # Only add edge from side 0 to avoid duplicates
                    edges.append((vi, index[other]))
    
    return total, edges

def _verify_girth(n, edges, min_girth):
    """Check if graph has girth >= min_girth using BFS from each vertex"""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS from each vertex to find shortest cycle
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[u] + 1 - dist[v] >= 2:
                    # Found a cycle of length dist[u] + 1 - dist[v] + 1
                    cycle_len = dist[u] + 1 - dist[v] + 1
                    if cycle_len < min_girth:
                        return cycle_len
    
    return min_girth  # All cycles >= min_girth

def _seed_construct(k, g):
    # Original seed code
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
