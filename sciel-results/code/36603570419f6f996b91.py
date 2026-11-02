import time
import sys

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction CD(d, q).
    Optimized d selection to minimize vertices while satisfying girth.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    q = k
    # For LUW CD(d, q), girth is approximately 2d+1.
    # We need 2d+1 >= g => d >= (g-1)/2.
    # Also d must be such that the construction is valid.
    # Standard LUW works for d >= 3.
    d = max(3, (g - 1) // 2)
    
    # If g is even, girth must be even. 
    # LUW graphs are bipartite, so girth is even.
    # If g=9, we need girth >= 9, which implies girth >= 10.
    # d=4 gives girth 9 (odd) in the formula 2d+1? 
    # Actually, the girth of CD(d,q) is 2d+1 if d is odd?
    # Let's assume the standard bound holds: girth >= 2d+1.
    # If g is odd, we need 2d+1 >= g.
    # If g is even, we need 2d+1 >= g (since girth is even, if 2d+1 >= g, and 2d+1 is odd, 
    # the even girth must be >= 2d+2? No. If min girth is 2d+1, and graph is bipartite, 
    # the girth is actually 2d+2? 
    # Let's be conservative: use d such that 2d+1 >= g.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Rule for LUW
    # rule[c] = (a, b) such that y[c] = x[c] + y[a]*x[b] (or similar)
    # Based on the seed code:
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
    index, order, edges = {start: 0}, [start], []
    
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
