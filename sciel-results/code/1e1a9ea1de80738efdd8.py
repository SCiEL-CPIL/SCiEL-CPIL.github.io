"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Strategy: Use the Lazebnik-Ustimenko-Woldar (LUW) construction CD(d,q) which gives
    explicit q-regular graphs with known girth. For k=3, g=16, we use q=3 and appropriate d.
    The LUW graph CD(d,q) has girth at least d+5 for odd d.
    
    For g=16: we need d+5 >= 16, so d >= 11. Use d=11 (odd).
    The number of vertices in the connected component is q^(d+1) = 3^12 = 531441... that's too many.
    
    Actually, let me reconsider. The LUW construction CD(d,q) where d is odd gives a graph
    with girth d+5. The number of vertices is q^(d+1). For q=3, d=11: 3^12 = 531441 vertices.
    
    But wait - the Moore bound for n(3,16) is 2*(3*(3^7-1)/(3-1)) = 2*3*(2187-1)/2 = 3*2186 = 6558.
    Actually the Moore bound formula for k-regular graphs with girth g:
    For even g: n >= k*(k-1)^(g/2-1) + (k-1)^(g/2-1) = (k+1)(k-1)^(g/2-1)
    For g=16, k=3: n >= 4*2^7 = 512. That matches the stated lower bound.
    
    The LUW graph is much larger than optimal but is valid. Let me use it as a fallback.
    
    For a better approach, I could try to find smaller constructions. But given the time
    constraint, let me implement the LUW construction correctly and efficiently.
    """
    start_time = time.time()
    DEADLINE = 120.0
    TIME_MARGIN = 10.0  # safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > DEADLINE - TIME_MARGIN
    
    # LUW construction CD(d, q)
    # For odd d, girth >= d + 5
    # We need girth >= g, so d >= g - 5
    # Choose smallest odd d >= g - 5
    q = k  # field size, must be prime or 4
    d = g - 5
    if d % 2 == 0:
        d += 1  # make it odd
    
    # For q=3, d=11: vertices = q^(d+1) = 3^12 = 531441
    # This is valid but large. Let's implement it.
    
    # GF(q) arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Rule for LUW: for c = 2..d, rule[c] = (a, b) such that
    # point p ~ line l iff l[c] - p[c] = l[a] * p[b]
    # The rule depends on d and follows a specific pattern.
    # From the LUW paper, for d odd:
    # rule[2] = (1, 1)
    # rule[3] = (2, 1) 
    # rule[4] = (1, 2)
    # and then periodic with period 4:
    # For c >= 5: let r = (c-5) % 4, c0 = c - r
    # rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        """Generate neighbors of a vertex in the LUW graph."""
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:  # point to line
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:  # line to point
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    # Build the connected component containing the zero vertex
    start_vertex = (0, (None,) + (0,) * d)
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    for vertex in order:
        if check_deadline():
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
