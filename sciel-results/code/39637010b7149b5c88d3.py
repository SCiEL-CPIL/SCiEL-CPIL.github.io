"""Construct a 5-regular graph with girth 9 (cage problem n(5,9))."""
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    # Use LUW construction with optimal d for the given girth
    n, edges = _luw_construct_optimal(k, g)
    
    return n, edges

def _luw_construct_optimal(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction: q-regular bipartite graph.
    The girth of LUW(q, d) is 2d + 2.
    We need 2d + 2 >= g, so d >= (g-2)/2.
    Since d must be an integer, d = ceil((g-2)/2).
    """
    q = k
    # For bipartite LUW, girth = 2d + 2
    # Need 2d + 2 >= g => d >= (g-2)/2
    d = (g - 2 + 1) // 2  # ceil((g-2)/2)
    if 2 * d + 2 < g:
        d += 1
    
    # For q=5 (prime), use arithmetic mod q
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: x * y % q
    
    # Rule for LUW: we need rules for c=2..d
    # Standard LUW rules for prime q:
    # c=2: (1,1)
    # c=3: (2,1)
    # c=4: (1,2)
    # For c>=5: periodic with period 4
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
