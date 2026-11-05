import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # L-U-W construction parameters
    # q must be prime or 4. k is the degree.
    # If k is not prime and not 4, we can't use standard L-U-W directly with q=k.
    # However, the task usually passes valid k for this family or expects us to handle it.
    # For k=5, q=5 is prime.
    # If k is not prime, we might need a different construction or fail.
    # The prompt implies k=5, g=9.
    
    q = k
    # Check if q is prime or 4. If not, this specific construction might not apply directly 
    # or we need a different q. But the interface fixes k.
    # We assume k is suitable for L-U-W (prime or 4).
    
    # Determine d
    # Girth of CD(d,q) is d+5.
    # We need d+5 >= g => d >= g-5.
    # Minimum d is 2.
    d = max(2, g - 5)
    
    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Assume q is prime
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Rule for L-U-W: 
    # The condition is l[c] - p[c] = l[a] * p[b]
    # The rule defines (a, b) for each c from 2 to d.
    # Standard L-U-W rules:
    # For c=2, a=1, b=1? No, indices are 1..d.
    # In the seed code, the rule was defined dynamically.
    # Let's use the explicit rule from the paper or the seed code's logic which was correct.
    # Seed code rule:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d+1):
    #    r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #    rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # This rule ensures the girth property.
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The indices in the tuple (a,b) refer to the 1-based indexing of the vector components?
        # In the seed code:
        # y[c] = add(x[c], mul(y[a], x[b]))
        # x and y are tuples of length d+1 (indices 0..d).
        # The rule values (a,b) are used as indices.
        # The seed code's rule generation seems to produce valid indices.
        candidates = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = candidates[r]

    def get_neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1. x[0] is None/side marker? 
        # In seed: start = (0, (None,) + (0,) * d)
        # So x has length d+1. x[1]..x[d] are the coordinates.
        for first in range(q):
            # first is the value of the 'first' coordinate of the neighbor?
            # In bipartite graph, point p (side 0) connects to line l (side 1).
            # The adjacency is defined by the rule.
            # The seed code iterates `first` in range(q).
            # This corresponds to fixing one coordinate of the neighbor.
            y = [None, first] + [0] * (d - 1)
            # Calculate remaining coordinates
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # p ~ l: l[c] - p[c] = l[a] * p[b]
                    # l[c] = p[c] + l[a] * p[b]
                    # Here x is p, y is l.
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # l ~ p: l[c] - p[c] = l[a] * p[b]
                    # p[c] = l[c] - l[a] * p[b]
                    # Here x is l, y is p.
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS
    for i in range(len(order)):
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (from side 0 to side 1)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
        if time.time() > deadline:
            break

    n = len(order)
    return n, edges
