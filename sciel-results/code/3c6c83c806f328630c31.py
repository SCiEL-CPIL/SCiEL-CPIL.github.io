import time

def construct(k, g):
    start_time = time.time()
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    q = k
    # For girth g, we need d such that girth(CD(d, q)) >= g
    # Girth is d+5 (odd d) or d+4 (even d)
    # We want girth >= 9.
    # If d is odd, d+5 >= 9 => d >= 4 (but d must be odd, so d >= 5).
    # If d is even, d+4 >= 9 => d >= 5 (but d must be even, so d >= 6).
    # So min d is 5.
    
    d = 5
    if (g - 5) % 2:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    # Ensure d is odd for girth d+5?
    # If d is even, girth is d+4.
    # If we want girth >= 9, and d is even, d+4 >= 9 => d >= 5 => d=6.
    # If d is odd, d+5 >= 9 => d >= 4 => d=5.
    # So d=5 is better (smaller n).
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
    index, order, edges = {start: 0}, [start], []
    
    # Time check
    if time.time() - start_time > 100:
        return len(order), edges # Should be empty or partial? 
        # Better to return valid small graph if possible?
        # LUW construction is fast.
        
    for vertex in order:
        if time.time() - start_time > 110:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
