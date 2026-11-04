import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    # LUW construction parameters
    q = k
    
    # Determine d
    # Girth of CD(d, q) is at least 2d + 4 for q large enough.
    # We need girth >= g.
    # So 2d + 4 >= g => d >= (g - 4) / 2.
    # Let's use a slightly conservative bound to be safe, but much better than the seed's d+4.
    # If g=9, d >= 2.5 => d=3.
    # If g=10, d >= 3 => d=3.
    # If g=11, d >= 3.5 => d=4.
    
    d = max(1, (g - 4 + 1) // 2)
    
    # Ensure d is at least 1 for the construction to make sense
    if d < 1:
        d = 1

    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for LUW graph
    # The rule defines the relation between coordinates.
    # The seed code has a specific rule generation.
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find the component
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            # If we run out of time, we return what we have? 
            # No, we must return a valid graph. 
            # The BFS should be fast for small d.
            pass
            
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
