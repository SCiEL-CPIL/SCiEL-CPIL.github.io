import numpy as np
import time

def construct(k, g):
    """
    Constructs a 3-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) method.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    q = k
    # d is the parameter for LUW. Girth is d+5 for odd d.
    # We need girth >= g.
    if (g - 5) % 2 == 0:
        d = g - 5
    else:
        d = g - 4
    if d < 3:
        d = 3

    # Precompute field operations
    if q == 2:
        # GF(2)
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: x & y
    elif q == 3:
        # GF(3)
        add = lambda x, y: (x + y) % 3
        sub = lambda x, y: (x - y) % 3
        mul = lambda x, y: (x * y) % 3
    else:
        # General prime
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Rule for LUW
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The rule depends on r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # BFS to find connected component
    # Vertices are (side, tuple_of_coords)
    # side 0: Points, side 1: Lines
    start = (0, (0,) * (d + 1)) # Index 0 is unused in seed? Seed used (None,) + (0,)*d
    # Seed: start = (0, (None,) + (0,) * d)
    # Let's stick to seed's representation to ensure correctness.
    start = (0, (None,) + (0,) * d)
    
    index = {start: 0}
    order = [start]
    edges = []
    
    # Optimization: Use a queue
    from collections import deque
    queue = deque([start])
    
    while queue:
        if time.time() > deadline:
            break
        v = queue.popleft()
        side, x = v
        # Generate neighbors
        # For each possible value of the first coordinate in the other side
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            # Calculate remaining coordinates
            valid = True
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point p ~ Line l: l[c] - p[c] = l[a] * p[b]
                    # l[c] = p[c] + l[a] * p[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line l ~ Point p: p[c] - l[c] = p[a] * l[b]
                    # p[c] = l[c] + p[a] * l[b]
                    # y[c] = x[c] + x[a] * y[b]
                    y[c] = add(x[c], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(other)
            
            # Add edge if from Point side to avoid duplicates
            if side == 0:
                edges.append((index[v], index[other]))

    return len(order), edges
