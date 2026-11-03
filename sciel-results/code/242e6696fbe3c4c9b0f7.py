import time

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (L-U-W) construction.
    The L-U-W graph CD(d, q) is q-regular and has girth 2d+1.
    We set q = k and d = (g-1)//2 to ensure girth >= g.
    """
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s limit
    
    q = k
    # For L-U-W, girth is 2d+1. We need 2d+1 >= g => d >= (g-1)/2.
    d = (g - 1) // 2
    
    # Ensure d is at least 2 for the construction to make sense
    if d < 2:
        d = 2

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Define the rule for L-U-W graph
    # The rule maps column c to coefficients (a, b) such that y[c] = y[a] * x[b] + x[c]
    # For CD(d, q), the rule is periodic with period 4?
    # Original code logic:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d + 1):
    #     r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #     rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The indices in the original code were:
        # r=0: (1, c0-2)
        # r=1: (c0-1, 1)
        # r=2: (c0, 1)
        # r=3: (1, c0+1)
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1? 
        # In original code: y = [None, first] + [0] * (d - 1)
        # So x has indices 1..d. Index 0 is None.
        # Total length is d+1.
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line: l[c] = p[c] + l[a] * p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point: p[c] = l[c] - l[a] * p[b] ? 
                    # Original: sub(x[c], mul(x[a], y[b]))
                    # Wait, the original code uses x for the current vertex and y for the new one.
                    # If side==0 (Point), x is Point, y is Line.
                    # If side==1 (Line), x is Line, y is Point.
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find connected component
    i = 0
    while i < len(order):
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
        # Time check
        if time.time() > deadline:
            break

    return len(order), edges
