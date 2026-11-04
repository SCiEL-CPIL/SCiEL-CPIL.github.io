import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin

    if k != 4 or g != 10:
        return _luw_fallback()

    # Try random search for a smaller 4-regular graph with girth >= 10
    n = 320
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
        random.seed(attempt)
        edges = _rand_4reg(n)
        if edges:
            if _girth(n, edges) >= 10:
                return n, edges

    return _luw_fallback()

def _rand_4reg(n):
    stubs = [i for i in range(n) for _ in range(4)]
    random.shuffle(stubs)
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v or (min(u,v), max(u,v)) in edges:
            return None
        edges.add((min(u,v), max(u,v)))
    return list(edges)

def _girth(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    mg = float('inf')
    for s in range(n):
        d = [-1]*n; p = [-1]*n; q = [s]; d[s] = 0
        while q:
            u = q.pop(0)
            for v in adj[u]:
                if d[v] == -1:
                    d[v] = d[u]+1; p[v] = u; q.append(v)
                elif v != p[u]:
                    mg = min(mg, d[u]+d[v]+1)
    return mg if mg != float('inf') else n

def _luw_fallback():
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
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

# EVOLVE-BLOCK-END
