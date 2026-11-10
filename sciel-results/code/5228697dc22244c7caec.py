import time
import random
from collections import deque

# EVOLVE-BLOCK-START
import numpy as np

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    # For girth g, we need 2d + 2 >= g => d >= (g-2)/2
    # The standard LUW uses d such that girth is 2d+2.
    # For g=10, d=4 gives girth 10.
    # The formula in the seed: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # If g=10: (10-5)%2 = 1 (odd). d = 10-5 = 5.
    # Wait, if d=5, girth is 12. n = 2 * 4^5 = 2048.
    # If d=4, girth is 10. n = 2 * 4^4 = 512.
    # The seed program calculates d=5 for g=10.
    # This is suboptimal! We should use d=4.
    
    # Correct d calculation:
    # We want girth >= g.
    # Girth of CD(d, q) is 2d + 2.
    # So 2d + 2 >= g => d >= (g-2)/2.
    # d must be integer.
    d = max(3, (g - 2 + 1) // 2) # ceil((g-2)/2)
    
    # For g=10: d = ceil(8/2) = 4.
    # For g=12: d = ceil(10/2) = 5.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

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

def construct(k, g):
    start_time = time.time()
    deadline = 140
    
    # Use the corrected d calculation
    best_n, best_edges = _luw_construct(k, g)
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
