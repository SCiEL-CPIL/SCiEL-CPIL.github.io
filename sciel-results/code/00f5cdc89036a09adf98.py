def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar construction.
    For k=3, we use q=3.
    The girth is at least d + 5.
    We choose d such that d + 5 >= g.
    """
    q = k
    # Ensure d is large enough for the girth
    # Girth >= d + 5
    # We want d + 5 >= g => d >= g - 5
    # The seed code logic:
    # d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # This ensures d is roughly g-5.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Field operations for GF(q)
    # q is 3 (prime)
    add = lambda x, y: (x + y) % q
    sub = lambda x, y: (x - y) % q
    mul = lambda x, y: (x * y) % q
    
    # Precompute the rules for the LUW construction
    # rule[c] = (a, b) such that y[c] = x[c] +/- y[a]*x[b]
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The 4 options for the rule
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1: (None, x1, ..., xd)
        # For points (side=0), x[0] is ignored.
        # For lines (side=1), x[0] is the first coordinate of the line.
        # The neighbours of a point p are lines l such that p is incident to l.
        # The neighbours of a line l are points p such that p is incident to l.
        
        # We iterate over the possible values of the first coordinate of the line/point?
        # In the LUW graph, the degree is q.
        # The neighbours are determined by the free variable in the system.
        # The seed code iterates `for first in range(q):`
        
        for first in range(q):
            # Initialize y
            y = [0] * (d + 1)
            y[0] = None
            y[1] = first
            
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

    # Start BFS from the zero point
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # Queue for BFS
    from collections import deque
    queue = deque([start])
    
    while queue:
        vertex = queue.popleft()
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(other)
            # Add edge only once (from point to line)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
