"""Construct a 5-regular graph with girth at least 9 (cage problem n(5,9)).
Uses the Lazebnik-Ustimenko-Woldar (LUW) construction for prime q=k.
For k=5, g=9: d=4 (since g-5=4 is even, use g-4=5? No: g=9, g-5=4 even -> d=5).
Actually: d = g - 5 if odd, else g - 4. 9-5=4 even -> d=9-4=5.
The LUW graph CD(q, d) has girth >= d+5 = 10 >= 9. It is q-regular.
For q=5, d=5, the connected component of zero has n vertices and is valid.
"""

def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(k, d) for prime k.
    # Girth is at least d + 5. We need girth >= g, so choose d such that d+5 >= g.
    # For odd d: girth >= d+5. For even d: girth >= d+5 as well (the construction works).
    # Actually the LUW paper: CD(d, q) has girth >= d+5 for odd d.
    # Let's use d = g - 5 if (g-5) is odd, else d = g - 4 (so d is odd and d+5 >= g).
    # For g=9: g-5=4 (even), so d=5 (odd), d+5=10>=9. Good.
    
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # GF(q) arithmetic for prime q
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return x * y % q
    
    # Rule for LUW construction: for c in 2..d, rule[c] = (a, b)
    # where the relation is y[c] = x[c] + y[a]*x[b] (for point-to-line)
    # or y[c] = x[c] - x[a]*y[b] (for line-to-point)
    # The rule depends on c mod 4:
    # c=2: (1,1), c=3: (2,1), c=4: (1,2)
    # For c >= 5: pattern repeats with period 4 starting at c=5
    # r = (c-5) % 4
    # c0 = c - (c-5)%4
    # r=0: (1, c0-2), r=1: (c0-1, 1), r=2: (c0, 1), r=3: (1, c0+1)
    
    rule = {}
    for c in range(2, d + 1):
        if c <= 4:
            rule[c] = {2: (1, 1), 3: (2, 1), 4: (1, 2)}[c]
        else:
            r = (c - 5) % 4
            c0 = c - r
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
    
    # BFS from the zero point to get the connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
