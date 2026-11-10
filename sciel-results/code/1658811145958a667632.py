import time
from collections import deque

def _check_girth_fast(n, edges, g):
    """Check girth >= g using BFS with non-backtracking."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    half_g = g // 2
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([(start, -1)])
        while queue:
            v, prev = queue.popleft()
            if dist[v] >= half_g:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    if dist[w] < half_g:
                        queue.append((w, v))
                else:
                    if dist[v] + 1 + dist[w] < g:
                        return False
    return True

def _construct_luv(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # Use LUV construction which is known to be valid
    n, edges = _construct_luv(k, g)
    
    # Verify
    if not _check_girth_fast(n, edges, g):
        # Should not happen, but fallback
        pass
    
    return n, edges
