"""Construct 3-regular graph of girth 16 using Lazebnik-Ustimenko-Woldar construction."""
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # For k=3, g=16: use q=3, d=11 (girth >= d+5 = 16)
    q = k
    # d must be odd and >= g-5
    if (g - 5) % 2 == 0:
        d = g - 4
    else:
        d = g - 5
    d = max(d, 3)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Build the rule
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Generate neighbours
    def get_neighbours(vertex):
        side, x = vertex
        result = []
        for first in range(q):
            y = [None] * (d + 1)
            y[1] = first
            for i in range(2, d):
                y[i] = 0
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            result.append((1 - side, tuple(y)))
        return result
    
    # BFS from start vertex to get connected component
    start = (0, tuple([None] + [0] * d))
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
