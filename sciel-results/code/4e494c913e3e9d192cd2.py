import numpy as np
from collections import deque

def construct(k, g):
    if k == 4 and g == 10:
        # Try circulant constructions first (deterministic, fast)
        result = try_circulant(4, 10)
        if result is not None:
            return result
        # Fallback: LUW construction (known 320-vertex graph)
        return luw_construct(4, 10)
    else:
        return luw_construct(k, g)

def try_circulant(k, g):
    """Search over circulant graphs C(n, {±a, ±b}) for girth >= g."""
    for n in range(244, 321, 2):
        half = n // 2
        for a in range(1, min(half, 60)):
            for b in range(a + 1, min(half, 60)):
                # Ensure 4 distinct offsets
                if a == n - a or b == n - b or a == n - b or n - a == b or n - a == n - b:
                    continue
                # Build adjacency
                adj = [[] for _ in range(n)]
                for i in range(n):
                    for d in (a, n - a, b, n - b):
                        j = (i + d) % n
                        adj[i].append(j)
                # Check girth
                if check_girth_fast(adj, g):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
    return None

def check_girth_fast(adj, g):
    """BFS-based girth check: returns True if girth >= g."""
    n = len(adj)
    for u in range(n):
        dist = [-1] * n
        dist[u] = 0
        q = deque([u])
        while q:
            v = q.popleft()
            for w in adj[v]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    q.append(w)
                else:
                    if dist[v] + dist[w] + 1 < g:
                        return False
    return True

def luw_construct(k, g):
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
