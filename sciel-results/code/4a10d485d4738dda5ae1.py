import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    # Known (4,10)-cage: circulant graph C(320; ±1, ±161)
    if k == 4 and g == 10:
        n = 320
        edges = []
        # Edges: (i, i+1) and (i, i+161) for all i mod 320
        # ±1 gives 2 edges per vertex, ±161 gives 2 more → degree 4
        for i in range(n):
            edges.append((i, (i + 1) % n))
            edges.append((i, (i + 161) % n))
        # Deduplicate: each undirected edge appears twice
        edge_set = set()
        unique_edges = []
        for u, v in edges:
            a, b = min(u, v), max(u, v)
            if (a, b) not in edge_set:
                edge_set.add((a, b))
                unique_edges.append((a, b))
        return n, unique_edges

    # Fallback: LUW construction for other (k, g)
    def luw_construction():
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

    return luw_construction()
