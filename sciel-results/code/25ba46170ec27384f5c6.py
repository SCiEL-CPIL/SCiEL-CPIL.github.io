import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    For k=3, g=16, this yields a graph with 2 * 3^11 vertices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    # LUW parameters
    q = k
    # Determine d such that girth >= g
    # Girth of CD(d, q) is at least d + 5 for odd d, d + 4 for even d?
    # Standard result: girth >= d + 5.
    # To be safe, we pick d such that d + 5 >= g => d >= g - 5.
    # If d is even, some sources say girth >= d + 4.
    # Let's stick to the logic in the seed which worked.
    if (g - 5) % 2 != 0:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

    # Field operations for GF(q)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # q is prime (k=3)
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Rule coefficients
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Precompute rules for speed
    rules = [None] * (d + 1)
    for c in range(2, d + 1):
        rules[c] = rule[c]

    def get_neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None, x[1] is the 'first' coord?
        # In the seed: y = [None, first] + [0] * (d - 1)
        # So x has length d+1.
        # x[0] is None. x[1] is the variable 'first' (0..q-1).
        # x[2..d] are the rest.
        
        # Optimization: Use lists for calculation, convert to tuple for key
        neighbors = []
        for first in range(q):
            # y is the line/point on the other side
            # y[0] is None
            # y[1] is 'first'
            # y[c] for c=2..d
            y = [None] * (d + 1)
            y[1] = first
            # y[2..d] initialized to 0? No, calculated.
            
            if side == 0:
                # Point -> Line
                # l[c] - p[c] = l[a] * p[b]  =>  l[c] = p[c] + l[a] * p[b]
                # Wait, the seed logic:
                # y[c] = add(x[c], mul(y[a], x[b])) if side == 0
                # x is the current vertex (Point). y is the neighbor (Line).
                # x[c] is p[c]. y[a] is l[a]. x[b] is p[b].
                # So y[c] = p[c] + l[a] * p[b].
                # We iterate 'first' which is y[1] = l[1].
                # But y[a] depends on a. If a=1, y[a]=first.
                # If a != 1, y[a] is determined by previous steps?
                # The loop in seed:
                # for c in range(2, d + 1):
                #   a, b = rule[c]
                #   y[c] = ... y[a] ...
                # This implies y[a] must be known.
                # Since c goes 2..d, and a is typically 1 or c-1 etc.
                # If a=1, y[1] is 'first', which is known.
                # If a > 1, y[a] must have been calculated in a previous iteration (a < c).
                # The rule construction ensures a < c?
                # rule[2]=(1,1). a=1.
                # rule[3]=(2,1). a=2. y[2] calculated in c=2.
                # rule[4]=(1,2). a=1.
                # rule[5]: c=5. r=0, c0=5. (1, 3). a=1.
                # It seems a is always 1 or c-1 or similar, so a < c.
                
                for c in range(2, d + 1):
                    a, b = rules[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                # Line -> Point
                # p[c] = l[c] - l[a] * p[b]  ?
                # Seed: y[c] = sub(x[c], mul(x[a], y[b])) if side == 1
                # x is Line. y is Point.
                # y[c] = l[c] - l[a] * p[b].
                # x[c]=l[c], x[a]=l[a], y[b]=p[b].
                for c in range(2, d + 1):
                    a, b = rules[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            
            neighbors.append((1 - side, tuple(y)))
        return neighbors

    # BFS to build the graph
    start = (0, (None,) + (0,) * d)
    # Note: start vertex x has x[1]=0.
    
    index = {start: 0}
    order = [start]
    edges = []
    
    # Use a queue for BFS
    queue = [0]
    head = 0
    
    while head < len(order):
        if time.time() > deadline:
            break
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        
        for u in get_neighbours(v):
            if u not in index:
                index[u] = len(order)
                order.append(u)
                queue.append(len(order) - 1)
            
            # Add edge only once (from point to line)
            if v[0] == 0:
                edges.append((v_idx, index[u]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
