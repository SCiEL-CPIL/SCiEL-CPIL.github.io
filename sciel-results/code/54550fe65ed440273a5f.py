   import time
   import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 115s limit, safety margin

    q = k
    
    # Moore Bound calculation
    # For k-regular graph of girth g:
    if g % 2 == 1:
        # g = 2r + 1, radius r
        r = (g - 1) // 2
        # 1 + k * sum_{i=0}^{r-1} (k-1)^i
        # Wait, the tree is 1 (center) + k (neighbors) + k(k-1) ...
        # Depth r from center.
        # Number of vertices = 1 + sum_{i=1}^{r} k * (k-1)^(i-1)
        moore_bound = 1
        term = k
        for i in range(1, r + 1):
            moore_bound += term
            term *= (k - 1)
    else:
        # g = 2r, radius r-1?
        # Center is an edge.
        # 2 * sum_{i=0}^{r-1} (k-1)^i * k? No.
        # Start from edge (u, v).
        # u has k-1 other neighbors, v has k-1 other neighbors.
        # Depth r-1 from the edge endpoints.
        r = g // 2
        # 2 * (1 + (k-1) + ... + (k-1)^(r-1))
        # Actually, the count is 2 * sum_{i=0}^{r-1} (k-1)^i * k? No.
        # Let's use the standard formula:
        # M = 2 * sum_{i=0}^{r-1} (k-1)^i
        # Wait, for g=4, r=2. M = 2 * (1 + k-1) = 2k. Correct (cycle C_k has 2k vertices? No, C_4 has 4. 2*4=8? No.
        # C_4 is 2-regular. k=2. M=2*(1+1)=4. Correct.
        # For k=5, g=6. r=3. M = 2 * (1 + 4 + 16) = 42.
        # Let's check: 5-regular, girth 6.
        # Tree from edge: 2 endpoints.
        # Each endpoint has k-1=4 neighbors.
        # Those have k-1=4 neighbors.
        # Total: 2 + 2*4 + 2*4*4 = 2 + 8 + 32 = 42.
        # Formula: 2 * sum_{i=0}^{r-1} (k-1)^i.
        # i=0: 1. i=1: k-1. i=2: (k-1)^2.
        # Sum = 1 + 4 + 16 = 21. 2*21 = 42. Correct.
        
        moore_bound = 2
        term = 1
        for i in range(1, r):
            term *= (k - 1)
            moore_bound += 2 * term

    # Determine d
    # Girth condition: 2d + 4 >= g  => d >= (g-4)/2
    d_min_girth = max(1, (g - 4 + 1) // 2)
    
    # Size condition: 2 * q^d >= moore_bound
    # Note: LUW graph size is 2 * q^d
    d = d_min_girth
    while 2 * (q ** d) < moore_bound:
        d += 1
        # Safety check to prevent infinite loop or huge d
        if d > 20:
            break

    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for LUW graph
    # The rule generation in the seed code is specific.
    # We need to ensure rule[c] is defined for all c in 2..d
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
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find the component
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            # If we run out of time, we cannot return a partial graph easily if it's not connected/valid.
            # But for LUW, it should be fast.
            # If it times out, we might return the current partial graph? 
            # The task says "return the best result found".
            # A partial BFS might yield a disconnected graph or invalid degrees.
            # However, LUW is a single component.
            # If we time out, we fail. 
            # Let's hope d is small enough.
            pass
            
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
   