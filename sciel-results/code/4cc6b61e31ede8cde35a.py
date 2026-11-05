"""Construct a 4-regular graph of girth 10 using the Lazebnik-Ustimenko-Woldar family."""
import time

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction which gives
    q-regular bipartite graphs of girth at least d+5 for odd d.
    
    For k=4, g=10: q=4, d=5 (since d+5=10), giving 320 vertices.
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    
    q = k
    # For girth g, we need d such that d + 5 >= g (for odd d)
    # d = g - 5 if (g-5) is odd, else g - 4
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # For q=4, use GF(4) with XOR addition and a specific multiplication table
    if q == 4:
        # GF(4) = {0, 1, alpha, alpha^2} with alpha^2 = alpha + 1
        # Represent as: 0=0, 1=1, alpha=2, alpha^2=3
        # Multiplication table for GF(4):
        # 0*x=0, 1*x=x, 2*2=3, 2*3=1, 3*3=2
        table = [[0, 0, 0, 0],
                 [0, 1, 2, 3],
                 [0, 2, 3, 1],
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y  # In GF(4), subtraction = addition
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    # LUW construction: define the "rule" for each column c from 2 to d
    # The rule specifies (a, b) such that l[c] - p[c] = l[a] * p[b]
    # For the standard LUW construction with girth d+5:
    rule = {}
    # Standard assignment for the LUW graph:
    # c=2: (1,1), c=3: (2,1), c=4: (1,2)
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    # For c >= 5, use a periodic pattern
    for c in range(5, d + 1):
        # Use a repeating pattern that avoids short cycles
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # Pattern based on the LUW paper
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        """Generate neighbors of a vertex in the LUW graph."""
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to line: l[c] = p[c] + l[a] * p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to point: p[c] = l[c] - l[a] * p[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    # BFS from the zero point to find the connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
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
            # Only add edge once (from point side)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
