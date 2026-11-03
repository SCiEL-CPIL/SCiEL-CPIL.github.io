import time

def construct(k, g):
    # Deadline guard: 150s wall-clock, leave 10s margin
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # For k=3, q=3.
    # Girth is d + 5 if d is odd, d + 6 if d is even?
    # The seed code uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=16: g-5=11 (odd), so d=11. Girth >= 11+5=16.
    
    q = k
    if q == 4:
        # GF(4) implementation
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # GF(q) for prime q
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Determine d
    # We want girth >= g.
    # LUW graph has girth d + 5 (if d odd) or d + 6 (if d even)?
    # Seed logic:
    # if (g - 5) % 2 == 0: d = g - 4 (even) -> girth d+6 = g+2 >= g
    # else: d = g - 5 (odd) -> girth d+5 = g >= g
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Precompute rules
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # BFS to construct the graph
    # Vertex representation: (side, tuple_of_coords)
    # side 0: points, side 1: lines
    # coords are length d+1, first is None for points? 
    # Seed: start = (0, (None,) + (0,) * d)
    # Neighbours of a point p:
    #   for first in range(q):
    #       y = [None, first] + [0] * (d - 1)
    #       for c in range(2, d + 1):
    #           a, b = rule[c]
    #           y[c] = add(p[c], mul(y[a], p[b]))
    #       yield (1, tuple(y))
    
    # This is O(q^d) which is 3^11 = 177147 vertices per side? 
    # Total 354294.
    # This is fast enough in Python if optimized.
    
    index = {}
    order = []
    edges = []
    
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)
    
    # Use a list for BFS queue
    queue = [start]
    head = 0
    
    while head < len(queue):
        if time.time() > deadline:
            break
            
        vertex = queue[head]
        head += 1
        side, x = vertex
        
        for first in range(q):
            # Build the neighbour
            # Optimization: Use a list for y, then tuple
            y = [None] * (d + 1)
            y[1] = first
            # y[0] is None
            # y[2..d] computed
            
            # We need to compute y[c] for c=2..d
            # y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
                    
            other = (1 - side, tuple(y))
            
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(other)
                
            # Add edge only once (from point side)
            if side == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
