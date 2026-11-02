import time
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0

    # LUW graph CD(d, q)
    # Girth is 2d + 2
    # We need 2d + 2 >= g => d >= (g - 2) / 2
    # We want smallest n = 2 * q^d, so smallest d.
    
    d = max(1, (g - 2) // 2)
    
    # Ensure d is at least 1
    # For g=10, d=4.
    # For g=6, d=2.
    
    q = k
    
    # GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Assume q is prime
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Rule for LUW graph
    # rule[c] = (a, b)
    # y[c] = y[a] * x[b] + x[c] (for points)
    # y[c] = x[c] - x[a] * y[b] (for lines)
    
    rule = {}
    # The standard LUW rule for c=2..d
    # For c=2: (1,1)
    # For c=3: (2,1)
    # For c=4: (1,2)
    # For c>=5: pattern
    # But we only need up to d.
    # If d < 2, we might have issues, but d>=1.
    # If d=1, no rule needed?
    # Let's handle d=1 separately or just ensure rule covers 2..d.
    
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            # x is a tuple of length d+1. x[0] is None, x[1..d] are values.
            # y[0] is None, y[1] is 'first', y[2..d] are computed.
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
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            break
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
