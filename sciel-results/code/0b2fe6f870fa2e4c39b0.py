"""Fast Lazebnik-Ustimenko-Woldar construction for cage graphs."""
import time
from collections import deque


def construct(k, g):
    """Construct a k-regular graph with girth >= g using the LW construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j).
    """
    deadline = time.time() + 115

    # Choose d for the LW construction.
    # Girth of LW(d, q): for odd d, girth = d+5; for even d, girth = d+4.
    # We want the smallest d with girth >= g.
    d = None
    for cand in range(3, 20):
        girth = cand + 5 if cand % 2 == 1 else cand + 4
        if girth >= g:
            d = cand
            break
    if d is None:
        d = 4
    q = k

    # Build the rule table
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # BFS from zero point
    # Vertex: (side, x_1, ..., x_d) where x_i in 0..q-1
    # side=0: point, side=1: line

    def get_neighbours(vertex):
        side, x = vertex
        x = list(x)
        result = []
        for first in range(q):
            y = [first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c - 1] = (x[c - 1] + y[a - 1] * x[b - 1]) % q
                else:
                    y[c - 1] = (x[c - 1] - x[a - 1] * y[b - 1]) % q
            result.append((1 - side, tuple(y)))
        return result

    start = (0, tuple([0] * (d + 1)))
    index = {start: 0}
    order = [start]
    adj = [[]]
    edges = []

    for i, vertex in enumerate(order):
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                adj.append([])
            j = index[other]
            if i < j:
                adj[i].append(j)
                adj[j].append(i)
                edges.append((i, j))

    n = len(order)
    return n, edges
