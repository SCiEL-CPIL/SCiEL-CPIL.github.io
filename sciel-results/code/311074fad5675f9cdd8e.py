"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    deadline = time.time() + 140  # safety margin before 150s kill
    best_n = None
    best_edges = None

    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned.
    q = k
    # We need girth >= g. CD(d,q) has girth >= d+5. So d+5 >= g => d >= g-5.
    # For g=9, d>=4. Try d=4 first (smallest, hence fewest vertices).
    d = g - 5
    if d < 3:
        d = 3
    # Ensure d is such that the construction works; for the rule we need d >= 2.
    # The rule dict covers up to d; extend if needed.

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    if len(order) > 0:
        best_n = len(order)
        best_edges = edges

    # Try a larger d if time permits (larger d => more vertices, so skip)
    # Instead, try to see if a different q helps? No, q must equal k for k-regular.

    # Return the best (only) result found
    if best_edges is None:
        # Fallback: trivial if something went wrong
        return 0, []

    return best_n, best_edges
# EVOLVE-BLOCK-END
