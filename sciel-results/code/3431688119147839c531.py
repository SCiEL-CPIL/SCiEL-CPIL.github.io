"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    q = k
    # For bipartite graphs, girth is even.
    # We need girth >= g.
    # If g is odd, we need girth >= g+1 (even).
    # If g is even, we need girth >= g.
    target_girth = g if g % 2 == 0 else g + 1
    
    # CD(d, q) has girth d + 5 (if d is odd) or d + 4 (if d is even)?
    # Actually, the girth of CD(d, q) is d + 5 for d >= 3.
    # Let's verify.
    # d=3 -> girth 8?
    # d=4 -> girth 9? No, bipartite.
    # The paper says: "The girth of CD(d, q) is d + 5."
    # So we need d + 5 >= target_girth.
    d = max(3, target_girth - 5)
    
    # Ensure d is such that the graph is valid.
    # If d < 3, it's not a good graph.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    # Rule for the LUW graph
    # The rule is defined for c = 2..d
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    # For c=2, (1,1)
    # For c=3, (2,1)
    # For c=4, (1,2)
    # For c > 4, the pattern repeats every 4?
    # The seed code used:
    # r, c0 = (c - 5) % 4, c - (c - 5) % 4
    # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    # This seems to be the standard LUW rule.
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1. x[0] is None (or 0), x[1]..x[d] are in GF(q)
        # Actually, the seed code uses x[0] as None, and x[1] as the first coordinate.
        # Let's stick to the seed's representation to avoid bugs.
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b]
                # side 0 is point, side 1 is line.
                # If vertex is point (side 0), we find line (side 1).
                # y is the line.
                # y[c] = x[c] + y[a] * x[b] ?
                # The seed code:
                # y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # BFS to find connected component
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

# EVOLVE-BLOCK-END
