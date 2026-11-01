import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, 10s margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    q = k
    
    # Calculate minimal d such that girth 2d+2 >= g
    # 2d + 2 >= g  =>  2d >= g - 2  =>  d >= (g - 2) / 2
    # d must be integer >= 3
    d = max(3, (g - 2 + 1) // 2)
    
    # If q is small, we might need to check if the construction is valid
    # For q=5, d=4, it is valid.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Rule for LUW construction
    # rule[c] = (a, b) such that y[c] = +/- y[a] * x[b]
    # For d=4:
    # c=2: rule[2] = (1, 1)
    # c=3: rule[3] = (2, 1)
    # c=4: rule[4] = (1, 2)
    # General formula from seed:
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            # y is a list of length d+1 (indices 0..d), but we use 1..d
            # x is a tuple of length d (indices 0..d-1 corresponding to 1..d)
            # Wait, in seed: x is tuple of length d.
            # y[c] uses x[c] which is index c-1?
            # Let's check seed indexing carefully.
            # Seed: y = [None, first] + [0] * (d - 1)
            # y[1] = first. y[2..d] computed.
            # x is passed as vertex[1].
            # In seed: x[c] is used.
            # If x is (x1, x2, ..., xd), then x[1] is x2?
            # Python tuples are 0-indexed.
            # Seed: start = (0, (None,) + (0,) * d)
            # This tuple has length d+1? (None, 0, ..., 0).
            # So x[0]=None, x[1]=0, ..., x[d]=0.
            # In neighbours:
            # y[c] = add(x[c], ...)
            # So x must have index c.
            # So x is 1-indexed effectively, with x[0] being dummy.
            
            y = [None, first] + [0] * (d - 1)
            # y[0]=None, y[1]=first, y[2..d]=0
            
            for c in range(2, d + 1):
                a, b = rule[c]
                # x[a] and x[b] must be valid indices.
                # a, b are in 1..d.
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
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
