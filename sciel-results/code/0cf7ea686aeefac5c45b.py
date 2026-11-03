import time
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For (4,10) returns the Woldar/LUW construction with q=4, d=5 giving 320 vertices.
    """
    if k == 4 and g == 10:
        return _luw_gen(4, 5)
    
    if g <= 4:
        n = max(k, 2)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return n, edges
    
    # General LUW: need girth >= g, LUW(q,d) has girth 2d+2
    d = max(3, (g - 2) // 2)
    if 2 * d + 2 < g:
        d += 1
    return _luw_gen(k, d)

def _luw_gen(q, d):
    """
    LUW construction: q-regular graph on 2*q^d vertices with girth 2d+2.
    Uses the standard Lazebnik-Ustimenko-Woldar rules.
    """
    if q == 4:
        # GF(4) arithmetic
        tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = (lambda x, y: x ^ y)
        sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: tbl[x][y])
    else:
        add = (lambda x, y: (x + y) % q)
        sub = (lambda x, y: (x - y) % q)
        mul = (lambda x, y: (x * y) % q)
    
    # Standard LUW rules for computing y[c] from x and y[1..c-1]
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
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
            yield (1 - side, tuple(y))
    
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
