import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    q = k
    # Minimal d such that girth 2d+2 >= g
    # d >= (g-2)/2
    d = max(2, (g - 2 + 1) // 2) # Ceiling division
    
    # If q == 4, use GF(4) logic, else GF(q)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Generate rules
    # The seed had specific rules for 2, 3, 4.
    # We need to ensure they are correct for the LUW construction.
    # The standard LUW rule is:
    # l[c] - p[c] = l[a] * p[b]
    # where a, b depend on c.
    # The seed's rule generation:
    rule = {}
    # Hardcoded for small d?
    # Let's use the seed's logic for rule generation but ensure it covers up to d.
    # The seed's logic:
    # rule[2] = (1, 1)
    # rule[3] = (2, 1)
    # rule[4] = (1, 2)
    # for c in 5..d:
    #   r, c0 = (c-5)%4, c - (c-5)%4
    #   rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    
    # This logic seems specific to the seed's implementation.
    # I will keep the rule generation as is, but ensure it runs up to d.
    
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
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
                    # point p ~ line l: l[c] - p[c] = l[a] * p[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p: l[c] - p[c] = l[a] * p[b]
                    # l[c] = p[c] + l[a] * p[b]
                    # x[c] = y[c] + x[a] * y[b]  => y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find connected component
    # For LUW, it's connected, so this will visit all 2*q^d vertices.
    # But we only add edges when side==0 to avoid duplicates.
    
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
                
        if time.time() > deadline:
            break
            
    # If we broke early, the graph might be incomplete or invalid.
    # However, for d=4, q=5, n=1250, it should be very fast.
    # 1250 vertices, 5 edges each. 3125 edge checks.
    # Neighbour generation is O(d*q).
    # It should finish in milliseconds.
    
    return len(order), edges
