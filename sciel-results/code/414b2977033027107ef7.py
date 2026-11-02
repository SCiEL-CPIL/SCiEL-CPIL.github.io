"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) family.
    For k=5, g=9, this yields a bipartite graph with girth 10 (which satisfies girth >= 9) 
    and approximately 500 vertices.
    """
    import time
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # LUW parameters
    # For odd girth g, we use d such that 2d+2 >= g.
    # If g is odd, girth must be even >= g, so girth >= g+1.
    # LUW girth is approx 2d+2.
    # So 2d+2 >= g+1 => 2d >= g-1 => d >= (g-1)/2.
    # For g=9, d >= 4.
    # The seed used d = g-5 = 4.
    
    q = k
    if q == 4:
        # GF(4) logic
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Determine d
    # We want girth >= g.
    # LUW graph CD(d, q) has girth 2d+2 (for large q).
    # We need 2d+2 >= g.
    # If g is odd, 2d+2 is even, so 2d+2 >= g+1.
    d = (g + 1) // 2
    # The seed used d = g-5 for g=9 -> d=4.
    # (9+1)//2 = 5. 
    # Wait. If d=5, girth is 12. n = 2*5^5 = 6250. Too big.
    # The seed used d=4. Girth 10. 10 >= 9. Valid.
    # So we should use the smallest d such that 2d+2 >= g.
    # For g=9, 2d+2 >= 9 => 2d >= 7 => d >= 3.5 => d=4.
    # So d=4 is correct.
    
    d = max(3, (g + 1) // 2 if (g % 2 == 1) else g // 2)
    # For g=9, d=4.
    
    # Rule definition for LUW
    # rule[c] = (a, b) such that y[c] = x[c] + y[a]*x[b]
    rule = {}
    for c in range(2, d + 1):
        # Standard LUW rule
        # For c=2, y[2] = x[2] + y[1]*x[1]? 
        # The seed used a specific rule. I will copy the seed's rule logic to ensure correctness.
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
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
    index = {start: 0}
    order = [start]
    edges = []
    
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
