import time
import sys
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LW) construction which gives explicit
    cages for small parameters.
    """
    # The LW construction gives a q-regular graph with girth at least 2*(ceil(g/2))-1
    # For k=5, g=9, q=5 gives a graph with girth at least 9.
    
    q = k
    # d is related to the girth. For bipartite graphs, girth >= 2d means
    # no cycles of length < 2d. We want girth >= g, so we need 2d >= g.
    # Actually, the LW construction for girth g uses d = ceil(g/2).
    # But the original code uses a more complex formula. Let's stick with a known good one.
    
    # For k=5, g=9:
    # The LW construction with q=5 and appropriate parameters.
    # d = 4 or 5?
    # Girth 9 means no cycles of length 3,4,5,6,7,8.
    # The LW construction with d=5 gives girth >= 2*5 = 10? Or 2*5-1 = 9?
    # Let's use the formula from the original: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=9: g-5 = 4, 4%2 == 0, so d = g-4 = 5.
    
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
    
    # Rule for the LW construction
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # The four options for the rule
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
    
    # Build the graph
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
