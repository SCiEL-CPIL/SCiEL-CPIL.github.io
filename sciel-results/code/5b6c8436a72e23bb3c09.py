"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # We choose d to be the smallest integer such that the girth is at least g.
    # The girth of CD(d, q) is typically 2d+1 or 2d+2.
    # For bipartite graphs, girth is even.
    # For g=9, d=4 yields girth 10 (or 9 if non-bipartite interpretation, but it is bipartite).
    # Using d=4 reduces n from 2*5^5=6250 to 2*5^4=1250.
    q = k
    
    # Determine d
    # If girth is 2d+2, we need 2d+2 >= g => d >= (g-2)/2
    # If girth is 2d, we need 2d >= g => d >= g/2
    # The seed used d=5 for g=9. 
    # Let's try d=4 for g=9.
    if g <= 6:
        d = 2
    elif g <= 8:
        d = 3
    elif g <= 10:
        d = 4
    elif g <= 12:
        d = 5
    else:
        d = (g - 2) // 2
        if d < 3: d = 3

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Rule definition for LUW
    # rule[c] = (a, b) such that y[c] = x[c] + y[a]*x[b] (for points)
    # Base cases
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
