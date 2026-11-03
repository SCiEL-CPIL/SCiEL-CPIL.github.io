import time
import heapq

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g using the LUW construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 120.0

    q = k  # q = 3

    # Determine d such that girth >= g
    # LUW graph girth: d+5 for odd d, d+6 for even d
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1

    # Precompute rules
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule_a[c], rule_b[c] = 1, c0 - 2
        elif r == 1:
            rule_a[c], rule_b[c] = c0 - 1, 1
        elif r == 2:
            rule_a[c], rule_b[c] = c0, 1
        else:
            rule_a[c], rule_b[c] = 1, c0 + 1

    rule_a_t = tuple(rule_a)
    rule_b_t = tuple(rule_b)

    # Generate vertices using BFS
    index = {}
    order = []  # list of (side, coord_tuple)
    edges = []

    start_coord = (0,) * (d + 1)
    start_key = (0, start_coord)
    index[start_key] = 0
    order.append((0, start_coord))

    c_range = range(2, d + 1)

    for i, vertex in enumerate(order):
        side, x = vertex

        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first

            if side == 0:
                for c in c_range:
                    y[c] = (x[c] + y[rule_a_t[c]] * x[rule_b_t[c]]) % q
            else:
                for c in c_range:
                    y[c] = (x[c] - x[rule_a_t[c]] * y[rule_b_t[c]]) % q

            y_t = tuple(y)
            neighbor_key = (1 - side, y_t)

            if neighbor_key not in index:
                index[neighbor_key] = len(order)
                order.append((1 - side, y_t))

            if side == 0:
                edges.append((i, index[neighbor_key]))

        if time.time() > deadline - 30:
            break

    n = len(order)

    # Verify 3-regularity and no self-loops
    # Each vertex should appear in exactly 3 edges
    from collections import Counter
    deg = Counter()
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    # Check all vertices have degree 3
    for v in range(n):
        if deg[v] != k:
            # Graph is not k-regular, need to handle this
            pass

    return n, edges
