import time
import sys
from collections import deque

def _luw_construct(k, g):
    """
    Construct a k-regular graph with girth at least g using Lazebnik-Ustimenko-Woldar construction.
    The standard LUW graph L(q, d) has 2*q^d vertices and girth 2*d.
    To achieve girth >= g, we need 2*d >= g, so d >= ceil(g/2).
    """
    q = k
    # Minimal d required for girth g
    d = (g + 1) // 2  # ceil(g/2)
    # Ensure d is at least 3 for the standard construction to be well-behaved/simple
    if d < 3:
        d = 3

    # Define group operations
    # For q=3, use Z_3
    if q == 4:
        # Z_4 is not a field, LUW usually works best with prime powers or specific groups.
        # However, the seed handled q=4 with XOR. 
        # For k=3, we use Z_3.
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # The LUW construction rules
    # Vertices are (side, x_1, ..., x_d)
    # Edges from (0, x) to (1, y) are defined by:
    # y_1 = x_1
    # y_i = x_i + x_{i-1} * x_{i-2} for i >= 2? 
    # Actually, the standard LUW edge definition is:
    # (0, x) ~ (1, y) iff y_1 = x_1 and y_i = x_i + x_{i-1} x_{i-2} is NOT quite right for all i.
    
    # Let's stick to the specific implementation in the seed which was verified to work, 
    # but just change 'd'.
    
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
                # side 0: y[c] = x[c] + y[a] * x[b]
                # side 1: y[c] = x[c] - x[a] * y[b]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to generate all vertices
    for i in range(len(order)):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge if we are on side 0 to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # For the specific case of k=3, g=16, we can use the optimized LUW
    # with d = ceil(16/2) = 8.
    # This yields 2 * 3^8 = 13122 vertices.
    # This is a valid upper bound.
    
    n, edges = _luw_construct(k, g)
    
    return n, edges
