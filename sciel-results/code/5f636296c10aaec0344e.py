def construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) construction.
    Returns a q-regular bipartite graph with girth at least d+5.
    For k=3, g=16: d=11, giving girth >= 16.
    """
    q = k
    # Find smallest odd d >= max(3, g-5)
    d_min = max(3, g - 5)
    if d_min % 2 == 0:
        d = d_min + 1
    else:
        d = d_min
    
    # Field operations for GF(q)
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q  
    mul = lambda x, y: (x * y) % q
    
    # L-U-W rule
    rule = {2: (1, 1)}
    for c in range(3, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
