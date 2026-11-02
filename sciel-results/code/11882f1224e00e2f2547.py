import time

def construct(k, g):
    """Construct a k-regular graph of girth at least g with as few vertices as possible.
    
    Uses the Lazebnik-Ustimenko-Woldar (LUW) graph CD(d, q) construction:
    - q = k (prime, or q=4)
    - d chosen so that the girth >= g
    - For odd girth g: need d+5 >= g, so d = g-5
    - For even girth g: need d+4 >= g, so d = g-4
    - Result: 2*q*(q+1) vertices, q-regular, girth d+5 (odd d) or d+4 (even d)
    """
    start_time = time.time()
    
    # Determine d for the LUW construction
    # LUW(d, q) has girth d+5 if d is odd, d+4 if d is even
    # We want girth >= g
    if (g - 5) % 2 == 1:
        # g is even: need d+4 >= g, d even, d = g-4
        d = g - 4
    else:
        # g is odd: need d+5 >= g, d odd, d = g-5
        d = g - 5
    
    # Ensure d >= 3 (minimum for LUW)
    d = max(3, d)
    
    q = k
    
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
    
    # Build the rule table for LUW
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
    
    # Build the graph via BFS from the zero vertex
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edges from points (side 0) to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
