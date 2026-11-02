import time

def construct(k, g):
    start_time = time.time()
    deadline = 100.0  # Safety margin

    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    # We optimize d to be the minimum required for the girth constraint.
    # The girth of CD(d, q) is generally considered to be 2d + 2.
    # We require girth >= g, so 2d + 2 >= g => d >= (g - 2) / 2.
    # d must be an integer >= 3.
    d = max(3, (g - 1) // 2)
    
    # If g is very small, d might be 3.
    # For g=9, d=4.
    # For g=10, d=4.
    # For g=11, d=5.

    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Generate the rule coefficients
    # The rule for level c is a pair (a, b) such that the relation is satisfied.
    # For LUW graphs, the rules follow a specific pattern.
    # The current code's rule generation:
    # rule[2] = (1, 1)
    # rule[3] = (2, 1)
    # rule[4] = (1, 2)
    # For c >= 5:
    # r, c0 = (c - 5) % 4, c - (c - 5) % 4
    # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None, x[1..d] are values
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    # l[c] - p[c] = l[a] * p[b]  => l[c] = p[c] + l[a] * p[b]
                    # Here y is the line, x is the point.
                    # y[c] = x[c] + y[a] * x[b]
                    # Note: y[a] might not be computed yet if a >= c?
                    # In LUW, the rule ensures a < c or similar causality?
                    # The standard LUW construction ensures that y[c] depends on y[a] where a < c.
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    # p[c] - l[c] = l[a] * p[b]  => p[c] = l[c] + l[a] * p[b]
                    # Here x is the line, y is the point.
                    # y[c] = x[c] + x[a] * y[b]
                    y[c] = add(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS
    for i in range(len(order)):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
        if time.time() > deadline:
            # If we run out of time, return what we have so far?
            # The graph might be incomplete.
            # But for n~1250, this should not happen.
            pass

    return len(order), edges
