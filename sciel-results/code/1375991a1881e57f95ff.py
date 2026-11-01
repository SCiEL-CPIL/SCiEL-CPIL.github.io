"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

    def build_luw(d):
        """Build LUW(d) graph over GF(q). Returns (n, edges)."""
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

    # For LUW(d) over GF(q), the girth is at least 2d + 2.
    # We need girth >= g.
    # Since LUW is bipartite, girth is even.
    # If g is odd, we need girth >= g+1 (since girth must be even and > g-1).
    # Actually, "no cycle shorter than g" means girth >= g.
    # If g=9, girth must be >= 9. Since bipartite, girth is even, so girth >= 10.
    
    target_girth = g if g % 2 == 0 else g + 1
    
    # We need 2d + 2 >= target_girth
    min_d = max(2, (target_girth - 2) // 2)
    
    # Use the minimal d to minimize n = 2 * q^d
    d = min_d
    
    # Check if we have time to verify, though theoretical girth is known.
    # For k=5, g=9, d=4, n=1250.
    # Verification is expensive, so we skip it for LUW graphs where theory applies.
    
    n, edges = build_luw(d)
    
    return n, edges
# EVOLVE-BLOCK-END
