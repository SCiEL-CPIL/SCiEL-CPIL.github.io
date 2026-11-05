"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # This is a q-regular bipartite graph.
    # The girth is 2*d + 2.
    # We require girth >= g.
    # 2*d + 2 >= g  =>  d >= (g - 2) / 2.
    # Minimal integer d is ceil((g - 2) / 2).
    # Also d must be at least 3 for the construction to be non-trivial/correct in this context.
    
    q = k
    if q < 3:
        # Handle small k if necessary, but task is likely k=5
        # For k=1, girth is infinite (tree)? No, simple graph.
        # For k=2, cycle.
        # We assume k >= 3 for this construction.
        pass
        
    # Calculate minimal d
    # d_min = ceil((g - 2) / 2)
    # (g - 2) / 2 = (g - 2 + 1) // 2 for integer ceil of x/2?
    # ceil(A/B) = (A + B - 1) // B
    # ceil((g-2)/2) = (g - 2 + 1) // 2 = (g - 1) // 2
    d = max(3, (g - 1) // 2)
    
    # Check if we have time to build
    if time.time() > deadline:
        # Fallback to a very small graph if we are already too late?
        # Or just return the best we can.
        # Since this is deterministic and fast, it should be fine.
        pass

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Define the rule for the LUW construction
    # The rule maps c to (a, b) such that y[c] = y[a] * x[b] or similar.
    # The original code had a specific rule generation.
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
                # The original code:
                # y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                # Note: x is a tuple of length d+1?
                # In original: y = [None, first] + [0]*(d-1). Length 1 + 1 + d - 1 = d+1.
                # Indices 0..d.
                # x[0] is None. x[1] is first. x[2..d] are values.
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # BFS to find the connected component
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
# EVOLVE-BLOCK-END
