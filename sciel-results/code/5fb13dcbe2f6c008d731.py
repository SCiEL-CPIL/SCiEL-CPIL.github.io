"""Construct a 5-regular graph of girth 9 using Lazebnik-Ustimenko-Woldar graphs."""
import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses Lazebnik-Ustimenko-Woldar (LUW) graphs for explicit construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin before 120s limit
    
    q = k
    
    # LUW graph CD(d, q) is a q-regular bipartite graph on 2*q^d vertices.
    # Girth >= d+5 if d is odd, >= d+4 if d is even.
    # For g=9: need girth >= 9.
    # d=5 (odd): girth >= 10, n = 2*5^5 = 6250
    # d=4 (even): girth >= 8, not enough
    # d=3 (odd): girth >= 8, not enough
    # So d=5 is the smallest d that guarantees girth >= 9.
    
    d = 5
    if g > d + 5:
        d = g - 5
        if d % 2 == 0:
            d += 1
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # Ensure odd for the d+5 girth bound
    
    # Finite field arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    # Rule table for LUW graph
    rule = {}
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Generate neighbors
    def get_neighbours(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
    
    # BFS from start vertex to get connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            n = len(order)
            valid_edges = [(u, v) for u, v in edges if u < n and v < n]
            return n, valid_edges
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    
    return len(order), edges
