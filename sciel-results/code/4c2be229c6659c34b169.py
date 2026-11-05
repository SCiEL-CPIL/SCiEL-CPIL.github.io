import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    q = k
    # For girth g, we need d such that d + 5 >= g
    # g is even (10), so d must be odd?
    # The LUW graph CD(d, q) has girth d+5 if d is odd?
    # Actually, the girth is at least d+5.
    # For g=10, d=5 gives girth >= 10.
    d = g - 5
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Precompute rules
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # BFS to generate the component
    # Vertex representation: (side, x0, ..., xd)
    # side 0: point, side 1: line
    # x is a tuple of length d+1
    
    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, x = vertex
        # Generate neighbors
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbor = (1 - side, tuple(y))
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            if side == 0:
                edges.append((i, index[neighbor]))
                
    return len(order), edges
