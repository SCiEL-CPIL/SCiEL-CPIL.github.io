"""Construct a (4,10)-cage graph with 320 vertices."""

def construct(k, g):
    """
    Construct a 4-regular graph with girth >= 10 on 320 vertices.
    Uses the Lazebnik-Ustimenko-Woldar construction with q=4, d=5.
    """
    import time
    start_time = time.time()
    deadline = start_time + 110

    if k == 4 and g == 10:
        q = 4
        d = 5  # girth >= d + 5 = 10

        # GF(4) operations
        gf4_add = lambda x, y: x ^ y
        gf4_mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        gf4_mul = lambda x, y: gf4_mul_table[x][y]
        gf4_sub = gf4_add  # In GF(4), subtraction = addition

        # LUW rule
        rule = {}
        for c in range(2, d + 1):
            if c % 2 == 0:
                rule[c] = (1, c - 2)
            else:
                rule[c] = (c - 1, 1)

        def luw_neighbours(side, coords):
            """Generate the 4 neighbours of a vertex."""
            result = []
            for first in range(q):
                other = [0] * (d + 1)
                other[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        # Point -> Line: l[c] = p[c] + l[a] * p[b]
                        other[c] = gf4_add(coords[c], gf4_mul(other[a], coords[b]))
                    else:
                        # Line -> Point: p[c] = l[c] - l[a] * p[b]
                        other[c] = gf4_sub(coords[c], gf4_mul(coords[a], other[b]))
                result.append((1 - side, tuple(other[1:])))
            return result

        # BFS from the zero point
        start = (0, tuple([0] * d))
        vertex_index = {start: 0}
        order = [start]
        edges = []

        for vertex in order:
            side, coords = vertex
            for other in luw_neighbours(side, coords):
                if other not in vertex_index:
                    vertex_index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((vertex_index[vertex], vertex_index[other]))

        n = len(order)
        return n, edges
    else:
        return 0, []
