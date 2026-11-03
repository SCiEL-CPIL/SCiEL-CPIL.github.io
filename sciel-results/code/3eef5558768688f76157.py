def construct(k, g):
    # Use Lazebnik-Ustimenko-Woldar (LUW) construction
    # q must be k for k-regularity
    q = k
    
    # Determine d such that girth >= g
    # Girth of CD(d, q) is at least 2d + 1
    # So we need 2d + 1 >= g  =>  d >= (g - 1) / 2
    d = (g - 1) // 2
    
    # Ensure d is at least 2 for the construction to be well-defined in this context
    if d < 2:
        d = 2

    # Define arithmetic for GF(q)
    # q is prime (k=5 is prime)
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: x * y % q
    
    # Define the rules for the LUW construction
    # rule[c] = (a, b) means y[c] depends on y[a] and x[b]
    # Standard rules for CD(d, q):
    # c=2: (1, 1)
    # c=3: (2, 1)
    # c=4: (1, 2)
    # c>=5: periodic pattern
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # Pattern from seed:
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
        for first in range(q):
            # y is a tuple of length d+1
            # y[0] is unused (None), y[1] is 'first', y[2..d] are calculated
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # Start from the zero point
    start = (0, (None,) + (0,) * d)
    
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only if we are on the "point" side to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
