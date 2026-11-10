import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # LUW Construction
    q = k
    # Determine d
    # For LUW, girth is d+5 if d is odd.
    # If g-5 is odd, d = g-5.
    # If g-5 is even, d = g-4 (which is odd).
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
        
    # Ensure d >= 3
    d = max(3, d)
    
    # Define field operations for GF(q)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Define the rule for LUW graph
    # rule[c] = (a, b) such that y[c] = y[a] * x[b] + x[c] (or similar)
    # Standard LUW rule:
    # c=2: (1, 1)
    # c=3: (2, 1)
    # c=4: (1, 2)
    # c=5: (1, 3) or (2, 2)?
    # The seed uses:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in 5..d:
    #   r, c0 = (c-5)%4, c - (c-5)%4
    #   rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    
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
                    # Point to Line
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
