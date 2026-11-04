"""Construct a 4-regular graph of girth 10 (cage problem n(4,10)).

Uses the Lazebnik-Ustimenko-Woldar (LUW) construction CD(d, q) with q = k.
For k=4, g=10: d=5 gives girth >= 10.
Returns the connected component of the origin.
"""
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Determine d: need d+5 >= g, d odd, d >= 3
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3

    q = k

    # GF(q) arithmetic
    if q == 4:
        mul_table = [[0,0,0,0],
                     [0,1,2,3],
                     [0,2,3,1],
                     [0,3,1,2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return mul_table[x][y]
    elif q == 2:
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return x & y
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return (x * y) % q

    # LUW rule
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    # BFS from origin to get connected component
    # Vertex: (side, coord_tuple) where coord_tuple has d elements (0-indexed)
    # side=0: point, side=1: line
    # Coordinates indexed 0..d-1 correspond to positions 1..d in the formula

    def get_neighbours(vertex):
        side, vals = vertex
        # vals is 0-indexed: vals[0] = coord_1, vals[1] = coord_2, ..., vals[d-1] = coord_d
        if side == 0:
            # Point, find lines
            for y1 in range(q):
                y = [0] * d
                y[0] = y1  # l_1
                for c in range(2, d + 1):
                    a, b = rule[c]
                    # l_c = p_c + l_{a_c} * p_{b_c}
                    # In 0-indexed: y[c-1] = vals[c-1] + y[a-1] * vals[b-1]
                    y[c - 1] = gf_add(vals[c - 1], gf_mul(y[a - 1], vals[b - 1]))
                yield (1, tuple(y))
        else:
            # Line, find points
            for x1 in range(q):
                x = [0] * d
                x[0] = x1  # p_1
                for c in range(2, d + 1):
                    a, b = rule[c]
                    # p_c = l_c - l_{a_c} * p_{b_c}
                    # In 0-indexed: x[c-1] = vals[c-1] - vals[a-1] * x[b-1]
                    x[c - 1] = gf_sub(vals[c - 1], gf_mul(vals[a - 1], x[b - 1]))
                yield (0, tuple(x))

    start_vertex = (0, tuple([0] * d))
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []

    for vertex in order:
        side, vals = vertex
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))

        if time.time() > deadline:
            break

    n = len(order)
    return n, edges
