"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) family.
    The LUW graph CD(d, q) is a q-regular bipartite graph with girth at least d + 5.
    We set q = k and d such that d + 5 >= g.
    Since the graph is bipartite, its girth is even. If g is odd, the girth will be at least g + 1.
    This satisfies the condition "no cycle shorter than g".
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    q = k
    # Determine d such that d + 5 >= g
    # The LUW graph has girth at least d + 5.
    # If g is odd, the actual girth (being even) will be at least g + 1, which is >= g.
    # If g is even, we need d + 5 >= g.
    # So d >= g - 5.
    # We choose the smallest d such that d + 5 >= g.
    # Also d must be at least 1.
    d = max(1, g - 5)
    
    # For the LUW construction to work as described in the seed, d is used in the vector length.
    # The seed code used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # Let's stick to the seed's logic for d to ensure compatibility with the 'rule' dictionary.
    # The rule dictionary in the seed handles c up to d.
    # If g=9, g-5=4. d=4.
    # If g=8, g-5=3. d=3.
    # The seed logic:
    # if (g-5) % 2 == 0 (g odd), d = g - 5.
    # if (g-5) % 2 == 1 (g even), d = g - 4.
    # Why?
    # If g is odd, say 9, d=4. Girth >= 9. Since bipartite, girth >= 10. 10 >= 9. OK.
    # If g is even, say 8, d=3. Girth >= 8. Since bipartite, girth >= 8. 8 >= 8. OK.
    # If we used d=4 for g=8, girth >= 9 -> 10. 10 >= 8. Also OK, but larger graph.
    # So the seed's choice of d minimizes the graph size for the LUW family.
    
    if (g - 5) % 2 == 0:
        d = g - 5
    else:
        d = g - 4
        
    d = max(1, d)

    # Define GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # q is prime (k is usually prime in these tasks, or we assume q=k is valid for GF)
        # If k is not prime, GF(k) does not exist. 
        # However, the task implies k is such that a solution exists.
        # For k=5, q=5 is prime.
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Precompute the rule for the LUW construction
    # The rule maps c to (a, b) for the equation y[c] = x[c] + y[a]*x[b] (or similar)
    # The seed code has a specific rule generation.
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            # y is a tuple of length d+1
            # y[0] is unused (None in seed), y[1] is 'first', y[2..d] are computed
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start_vertex = (0, (None,) + (0,) * d)
    index = {start_vertex: 0}
    order = [start_vertex]
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

    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
