"""Construct a k-regular graph with girth >= g using Lazebnik-Ustimenko-Woldar family."""
import time
from itertools import product

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction CD(q, d).
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s

    q = k
    # d is chosen so that girth >= d + 5 for odd d, or d + 4 for even d
    # We want girth >= g, so pick d such that d + 5 >= g (for odd d)
    # For the LUW construction with q=k, the girth is at least d+5 for odd d
    if (g - 5) % 2 == 0:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

    # GF(q) operations
    if q == 4:
        # GF(4) = GF(2)[x]/(x^2 + x + 1)
        # Elements: 0, 1, alpha, alpha+1 where alpha^2 + alpha + 1 = 0
        # alpha^2 = alpha + 1, alpha^3 = 1
        # Represent elements as 0,1,2,3 where 2=alpha, 3=alpha+1
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def gf_add(x, y):
            return x ^ y
        def gf_sub(x, y):
            return x ^ y
        def gf_mul(x, y):
            return mul_table[x][y]
    else:
        def gf_add(x, y):
            return (x + y) % q
        def gf_sub(x, y):
            return (x - y) % q
        def gf_mul(x, y):
            return x * y % q

    # Rule for LUW construction
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def compute_neighbours(vertex):
        """Given a vertex (side, coords), compute its q neighbours."""
        side, coords = vertex
        neighbours = []
        for first in range(q):
            y = [None] * (d + 1)
            y[0] = None
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point to line: y[c] = x[c] + y[a] * x[b]
                    y[c] = gf_add(coords[c], gf_mul(y[a], coords[b]))
                else:
                    # line to point: y[c] = x[c] - x[a] * y[b]
                    y[c] = gf_sub(coords[c], gf_mul(coords[a], y[b]))
            neighbours.append((1 - side, tuple(y[1:])))
        return neighbours

    # Build graph by BFS from origin
    start_vertex = (0, (0,) * d)
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []

    idx = 0
    while idx < len(order):
        if time.time() > deadline:
            break
        vertex = order[idx]
        side, coords = vertex
        for other in compute_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
        idx += 1

    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
