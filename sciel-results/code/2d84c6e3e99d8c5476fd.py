"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar
    construction CD(d, q) with q = k (or q = 4 for k=4). Returns the connected component
    containing the origin, which is a valid k-regular graph of girth >= g.
    """
    deadline = time.time() + 140  # Safety margin before 150s limit

    q = k
    if q == 4:
        # Special case for GF(4)
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Determine d such that girth >= d + 5 >= g, with d odd
    if (g - 5) % 2 == 1:
        d = max(3, g - 5)
    else:
        d = max(3, g - 4)

    # Precompute the rule: rule[c] = (a, b) meaning
    # for side 0: y[c] = x[c] + y[a] * x[b]
    # for side 1: y[c] = x[c] - x[a] * y[b]
    rule = [None, None, (1, 1), (2, 1), (1, 2)]
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule.append((1, c0 - 2))
        elif r == 1:
            rule.append((c0 - 1, 1))
        elif r == 2:
            rule.append((c0, 1))
        else:
            rule.append((1, c0 + 1))

    # BFS from the origin to build the connected component
    # Vertex representation: (side, tuple of coordinates)
    # side 0: point (x_1, ..., x_d), side 1: line (x_1, ..., x_d)
    # For side 0, x[0] is unused (index 0), coordinates are x[1]..x[d]
    # For side 1, x[0] is unused, coordinates are x[1]..x[d]
    # The origin is (0, (None, 0, 0, ..., 0))

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    # Use a queue for BFS to ensure we process in order
    queue = deque([0])

    while queue:
        if time.time() > deadline:
            break
        i = queue.popleft()
        vertex = order[i]
        side, x = vertex

        for first in range(q):
            # Compute the neighbor on the other side
            y_list = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y_list[c] = add(x[c], mul(y_list[a], x[b]))
                else:
                    y_list[c] = sub(x[c], mul(x[a], y_list[b]))
            other = (1 - side, tuple(y_list))

            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)

            # Add edge from side-0 vertices to avoid duplicates
            if side == 0:
                edges.append((i, index[other]))

    return len(order), edges
# EVOLVE-BLOCK-END
