import time
import collections
import random

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _construct_luw(k, g):
    """
    Construct a Lazebnik-Ustimenko-Woldar (LUW) graph.
    This graph is k-regular and has large girth.
    For k=3, g=16, this yields the standard LUW graph with n = 2 * 3^11 = 354294.
    """
    q = k
    # Determine d such that the girth is at least g.
    # Girth of LUW(q, d) is 2 * (d + 1) if q is odd? 
    # Actually, the girth is known to be at least 2d.
    # For k=3, g=16, we need 2d >= 16 => d >= 8.
    # The seed used d=11. Let's stick to the seed's logic to ensure girth >= 16.
    # The seed logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=16, g-5=11 (odd), so d=11.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Rule for the LUW graph
    # rule[c] = (a, b) such that y[c] = op(x[c], y[a] * x[b])
    # For c=2, rule is (1, 1)
    # For c=3, rule is (2, 1)
    # For c=4, rule is (1, 2)
    # General rule for c > 4:
    # r, c0 = (c - 5) % 4, c - (c - 5) % 4
    # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            # x is a tuple of length d+1, x[0] is unused (None)
            # y is calculated
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
    
    # BFS to enumerate all vertices and edges
    # Since the graph is regular and connected, this will find all 2*q^d vertices
    for i in range(len(order)):
        if _time_left() < 5.0:
            break
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once. 
            # Since the graph is bipartite, we can just add edge from side 0 to side 1.
            # Or simply, if vertex[0] == 0, add edge.
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # For k=3, g=16, the LUW graph is the standard fallback.
    # The random search is too slow and unlikely to succeed.
    # We directly construct the LUW graph.
    
    n, edges = _construct_luw(k, g)
    
    return n, edges
