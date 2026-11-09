import time
import random
from collections import deque

_DEADLINE = None

def _time_left(margin=5.0):
    if _DEADLINE is None:
        return float("inf")
    return _DEADLINE - time.monotonic() - margin

def _girth_ok(adj, n, g):
    # Check for cycles of length < g
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    dq.append(v)
                else:
                    if v != parent[u] and u != parent[v]:
                        if dist[u] + dist[v] + 1 < g:
                            return False
    return True

def _random_regular_graph(n, k, max_attempts=50):
    if (n * k) % 2 != 0:
        return None
    for _ in range(max_attempts):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
            edges.add((u, v))
        
        if valid:
            adj = [[] for _ in range(n)]
            edge_list = []
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                edge_list.append((u, v))
            return adj, edge_list
    return None

def _luw_component(q, d):
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

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
    index, order = {start: 0}, [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # Fallback: LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if d + 5 < g:
        d += 1
    n_luw, edges_luw = _luw_component(q, d)
    best_n = n_luw
    best_edges = edges_luw
    
    # Moore bound
    r = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** e for e in range(r))
    
    # Search for smaller n
    # Start from a bit below LUW
    n_try = n_luw - 50
    
    while n_try > moore and _time_left() > 0:
        res = _random_regular_graph(n_try, k)
        if res:
            adj, edges = res
            if _girth_ok(adj, n_try, g):
                best_n = n_try
                best_edges = edges
                # Try to go lower
                n_try -= 1
                continue
        n_try -= 1
        
    return best_n, best_edges
