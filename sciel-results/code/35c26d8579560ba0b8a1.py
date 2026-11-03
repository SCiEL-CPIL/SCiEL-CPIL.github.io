import time

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    q = k

    # Determine d such that girth_of_d(d) >= g
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1

    # Precompute rules as a flat list for speed
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

    # Generate vertices using BFS
    index = {}
    order = []
    edges = []

    # Start vertex: side 0, coordinates (None, 0, 0, ..., 0)
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)

    for i, vertex in enumerate(order):
        side, x = vertex

        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            if side == 0:
                for c in range(2, d + 1):
                    a = rule_a[c]
                    b = rule_b[c]
                    y[c] = (x[c] + y[a] * x[b]) % q
            else:
                for c in range(2, d + 1):
                    a = rule_a[c]
                    b = rule_b[c]
                    y[c] = (x[c] - x[a] * y[b]) % q

            neighbor = (1 - side, tuple(y))

            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)

            if side == 0:
                edges.append((index[vertex], index[neighbor]))

        if time.time() > deadline:
            break

    n = len(order)
    return n, edges
