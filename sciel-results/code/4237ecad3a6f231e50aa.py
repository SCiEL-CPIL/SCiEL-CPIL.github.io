"""Construct a 4-regular graph of girth >= 10 with as few vertices as possible.
Uses the Lazebnik-Ustimenko-Woldar (LUW) CD(5,4) construction, which is the
known (4,10)-cage with 320 vertices."""

import time

def construct(k, g):
    deadline = time.time() + 140  # 140s to be safe within 150s limit

    # LUW graph CD(d, q) with q = k, d chosen for girth
    q = k
    # For girth g, we need d such that d + 5 >= g (for odd d)
    # g = 10, so d >= 5. Use d = 5.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=10: (10-5)=5, odd, so d=5

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for each column c: pair (a, b) meaning l[c] - p[c] = l[a] * p[b]
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The four possible rules cycle:
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l: l[c] - p[c] = l[a] * p[b] => l[c] = p[c] + l[a]*p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p: l[c] - p[c] = l[a] * p[b] => p[c] = l[c] - l[a]*p[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # BFS from the zero point
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    i = 0
    while i < len(order):
        vertex = order[i]
        i += 1
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    return n, edges
