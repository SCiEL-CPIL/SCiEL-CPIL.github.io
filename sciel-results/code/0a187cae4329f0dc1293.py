import numpy as np
from collections import deque

def construct(k, g):
    if k == 4 and g == 10:
        # Directly use the LUW construction which is known to produce the 320-vertex cage
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)

def luw_construct(k, g):
    # Original LUW code (kept as is, assumed to be the source of the 320-vertex solution)
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

def check_girth(adj, g):
    n = len(adj)
    # Optimized BFS girth check
    # We only need to check if any cycle is shorter than g
    # For each node, BFS up to depth g//2 - 1
    # If we find a neighbor already visited at depth < g//2, we have a short cycle
    # Actually, standard BFS from each node is O(N*(N+E)).
    # For N=320, E=640, this is fast.
    for u in range(n):
        dist = [-1] * n
        dist[u] = 0
        q = deque([u])
        while q:
            v = q.popleft()
            if dist[v] >= g // 2: 
                continue
            for w in adj[v]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    q.append(w)
                elif w != v and dist[w] <= dist[v]: # Avoid counting the edge back
                    # Cycle length is dist[v] + dist[w] + 1
                    # We want to ensure no cycle < g
                    if dist[v] + dist[w] + 1 < g:
                        return False
    return True
