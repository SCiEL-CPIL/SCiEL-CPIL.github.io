import time

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # For k=4, we use the Lazebnik-Ustimenko-Woldar graph CD(d, 4)
    # This graph is 4-regular.
    # We need girth >= g.
    # Girth of CD(d, q) is at least 2d + 4.
    # So we need 2d + 4 >= g => d >= (g - 4) / 2.
    # Minimum d is 3 for the standard construction to be well-defined and efficient.
    
    if k != 4:
        # Fallback for other k (not expected in this specific task, but for safety)
        # Use a simple cycle if possible, or just return a dummy.
        # But the task specifies k=4.
        pass

    q = k  # Must be 4 for 4-regularity in this family
    
    # Calculate required d
    min_d = (g - 4 + 1) // 2 if (g - 4) % 2 else (g - 4) // 2
    if min_d < 3:
        min_d = 3
    d = min_d
    
    # Finite field GF(q) operations
    # For q=4, we use the specific table.
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # General prime q
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
    # Rule for LUW graph
    # The rules for CD(d, q) are defined for c = 2 to d.
    # Standard rules for d >= 3:
    # c=2: (1, 1)
    # c=3: (2, 1)
    # c=4: (1, 2)
    # c=5: (1, 3)
    # c=6: (5, 1) ? No, the pattern continues.
    # The seed used a specific pattern.
    # For d=3, we only need rules for c=2, 3.
    # Wait, the vertex is a tuple of length d+1.
    # x = (x_0, x_1, ..., x_d).
    # y is computed from x.
    # y[0] is None. y[1] is the free variable.
    # y[c] depends on previous y and x.
    # For d=3:
    # y[2] = y[1]*x[1] + x[2]  (rule 2: (1,1))
    # y[3] = y[2]*x[1] + x[3]  (rule 3: (2,1))
    # This yields a graph.
    
    rule = {}
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            # General pattern from seed
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
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS
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
