import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    # For k=3, q=3, d=11 gives girth 16 and n = 2 * 3^11 = 531442 vertices
    q = k
    d = g - 5  # For g=16, d=11

    # Precompute multiplication table for GF(q) when q is prime
    # For q=3: 0*0=0, 0*1=0, 0*2=0, 1*0=0, 1*1=1, 1*2=2, 2*0=0, 2*1=2, 2*2=1
    if q == 3:
        mul_table = [
            [0, 0, 0],
            [0, 1, 2],
            [0, 2, 1]
        ]
        def add_mod(x, y):
            r = x + y
            return r - q if r >= q else r
        def sub_mod(x, y):
            r = x - y
            return r + q if r < 0 else r
        def mul_mod(x, y):
            return mul_table[x][y]
    else:
        def add_mod(x, y):
            r = x + y
            return r - q if r >= q else r
        def sub_mod(x, y):
            r = x - y
            return r + q if r < 0 else r
        def mul_mod(x, y):
            return (x * y) % q

    # Precompute the rule for each c from 2 to d
    # rule[c] = (a, b) such that for points: l[c] = p[c] + l[a]*p[b]
    # For lines: p[c] = l[c] - l[a]*p[b]
    rule = {}
    for c in range(2, d + 1):
        if c % 2 == 0:
            # c = 2j, use a = 2j-1, b = 2j-2
            a = c - 1
            b = c - 2
        else:
            # c = 2j+1, use a = 2j, b = 2j-1
            a = c - 1
            b = c - 2
        rule[c] = (a, b)

    # Vertex encoding: (side, x0, x1, ..., xd)
    # side 0: point, side 1: line
    # x0 is always 0 (dummy)
    # We encode as a single integer for fast hashing
    # side + 1 + sum(x_i * q^(i-1)) for i=1..d
    # Actually, let's use tuples for clarity first, then optimize if needed

    def get_neighbors(vertex):
        side, x = vertex
        # x is a tuple of length d+1: (x0, x1, ..., xd)
        # x0 is always 0
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l
                    # l[c] = p[c] + l[a] * p[b]
                    y[c] = add_mod(x[c], mul_mod(y[a], x[b]))
                else:
                    # line l ~ point p
                    # p[c] = l[c] - l[a] * p[b]
                    y[c] = sub_mod(x[c], mul_mod(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, tuple([0] + [0] * d))
    index = {start: 0}
    order = [start]
    edges = []

    # BFS
    i = 0
    while i < len(order):
        if time.time() > deadline:
            break

        vertex = order[i]
        i += 1
        for other in get_neighbors(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
