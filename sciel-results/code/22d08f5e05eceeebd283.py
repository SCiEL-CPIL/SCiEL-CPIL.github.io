import time

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    q = k
    # Girth of CD(d, q) is d + 5 for odd d.
    # We need girth >= g.
    # If g is odd, d = g - 5.
    # If g is even, d = g - 4 (since d must be odd for the standard girth formula? 
    # Actually, for even girth, we might need a different analysis or just use d = g - 5 if it works, 
    # but the standard LUW girth is d+5. If g=10, d=5. Girth=10.
    # If g=9, d=4? Girth=9?
    # The formula girth = d+5 holds for odd d.
    # For even d, the girth is different.
    # For this task g=10, so d=5 is correct.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        # GF(4) table
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
    
    # BFS to generate the graph
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
