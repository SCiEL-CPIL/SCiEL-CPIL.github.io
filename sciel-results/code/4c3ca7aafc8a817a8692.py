"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # We choose d to minimize n = q^(d+1) subject to girth >= g.
    q = k
    
    # Determine the best d
    best_d = None
    best_n = float('inf')
    
    # d must be at least 2 for the construction to make sense in this context
    # girth for odd d is d+5, for even d is d+6
    for d in range(2, 20):
        if d % 2 == 1:
            girth = d + 5
        else:
            girth = d + 6
        
        if girth >= g:
            n = q ** (d + 1)
            if n < best_n:
                best_n = n
                best_d = d
    
    if best_d is None:
        # Fallback to a large d
        best_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
    d = best_d
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Rule for LUW graph
    # The rule definition in the seed seems specific to the implementation of the recurrence.
    # We must ensure the rule is correct for the chosen d.
    # The seed's rule generation:
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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
# EVOLVE-BLOCK-END
