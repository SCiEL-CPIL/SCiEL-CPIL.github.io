import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Determine d for LUW construction
    # Girth of LUW(d, q) is 2d. We need girth >= g.
    # So 2d >= g => d >= ceil(g/2).
    # Also d >= 3 for generalized polygons to be well-defined in this context?
    # LUW usually works for d >= 3.
    d = max(3, (g + 1) // 2)
    
    q = k
    
    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Rule for LUW construction
    # rule[c] = (a, b) such that y[c] = x[c] +/- y[a] * x[b]
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Precompute rules for faster access
    rules = [None] * (d + 1)
    for c in range(2, d + 1):
        rules[c] = rule[c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            if side == 0:
                # Point to Line
                for c in range(2, d + 1):
                    a, b = rules[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                # Line to Point
                for c in range(2, d + 1):
                    a, b = rules[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find all vertices and edges
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, x = vertex
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (from point side)
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
