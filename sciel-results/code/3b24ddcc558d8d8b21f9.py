"""Constructs a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction."""

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using the LUW construction CD(d, q).
    The graph is bipartite with 2 * q^d vertices.
    """
    q = k
    # Determine d such that the girth is at least g.
    # For LUW, girth is d + 5 if d is odd, d + 4 if d is even.
    # We want girth >= g.
    if g <= 7:
        # Small girths might be handled differently, but LUW works for d>=3.
        # For g=7, d=3 gives girth 8. d=2 gives girth 6?
        # Let's stick to the formula in the original code.
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
    # Ensure d is large enough
    if d < 3:
        d = 3

    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Define the rule for LUW construction
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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
