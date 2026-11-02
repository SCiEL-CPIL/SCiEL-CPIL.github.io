"""Improved LZW construction for cage graphs with deadline guard."""
import time

def construct(k, g):
    deadline = time.time() + 140
    q = k
    # LZW CD(d,q): q-regular bipartite graph, girth >= 2d+4
    # Need girth >= g: 2d+4 >= g => d >= ceil((g-4)/2), min d=3
    d = max(3, (g - 4 + 1) // 2)
    while 2 * d + 4 < g:
        d += 1

    # GF(q) arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Standard LZW rules: for c=2,...,d, define (a_c, b_c)
    # c=2: (1,1), c=3: (2,1), c=4: (1,2), c=5: (4,1), c=6: (1,5), c=7: (5,1), c=8: (1,6), ...
    # Pattern: for c >= 5, cycles through (c-1,1), (1,c), (c-1,1) pattern... 
    # Actually the standard: rule[c] alternates between (c-1, 1) and (1, c-1) starting from c=5
    # Let me use the known correct pattern:
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    # For c >= 5: the pattern is (c-1, 1) for odd offset, (1, c-1) for even offset from 5
    # Standard LZW: for c=5: (4,1), c=6: (1,5), c=7: (6,1), c=8: (1,7), ...
    # i.e., for c>=5: if (c-5) is even, rule=(c-1,1); if odd, rule=(1,c-1)
    for c in range(5, d + 1):
        if (c - 5) % 2 == 0:
            rule[c] = (c - 1, 1)
        else:
            rule[c] = (1, c - 1)

    # BFS: vertices are (side, coords_tuple) where coords has d elements (indices 0..d-1 = x_1..x_d)
    # side=0: point, side=1: line
    
    def compute_neighbours(vertex):
        side, x = vertex  # x is tuple of d elements, x[i] = x_{i+1}
        result = []
        for first in range(q):
            y = [0] * d
            y[0] = first  # y_1 = first
            for c in range(2, d + 1):
                # y[c] corresponds to y[c-1] in 0-indexed
                a, b = rule[c]  # a, b are 1-indexed
                if side == 0:  # point p to line l: l_c = p_c + l_a * p_b
                    y[c - 1] = add(x[c - 1], mul(y[a - 1], x[b - 1]))
                else:  # line l to point p: p_c = l_c - l_a * p_b
                    y[c - 1] = sub(x[c - 1], mul(x[a - 1], y[b - 1]))
            result.append((1 - side, tuple(y)))
        return result

    start = (0, tuple([0] * d))
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in compute_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    return n, edges
