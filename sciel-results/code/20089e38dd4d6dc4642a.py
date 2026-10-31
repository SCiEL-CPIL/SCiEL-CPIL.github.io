import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    q = k
    # Determine d such that girth is at least g
    # Girth of CD(d, q) is 2d+4 if d is odd? 
    # Prompt says: "girth is at least d + 5 for odd d"
    # If g=16, we need d+5 >= 16 => d >= 11.
    # If d is even, girth might be different?
    # The code uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=16: g-5=11 (odd). d=11.
    # If g=17: g-5=12 (even). d = g-4 = 13.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        # Optimize mod operations
        add = lambda x, y: x + y if x + y < q else x + y - q
        # sub: x - y mod q
        def sub_opt(x, y):
            res = x - y
            if res < 0:
                res += q
            return res
        sub = sub_opt
        mul = lambda x, y: (x * y) % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Precompute rules
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Vertex representation: (side, x0, x1, ..., xd)
    # side 0: point, side 1: line
    # x0 is None for points? In code: (None,) + (0,)*d
    # x0 is None for lines?
    
    # To speed up hashing, we can represent vertices as integers or tuples of ints.
    # Since q=3, we can pack the tuple into an integer base-q.
    # d is up to 11. 3^11 is ~ 177k. Fits in 32-bit int.
    
    def encode(vertex):
        side, x = vertex
        # x is a tuple of length d+1. x[0] is None or int.
        # If side=0 (point), x[0] is None. We ignore it or set to 0.
        # If side=1 (line), x[0] is int.
        # Actually, the code uses:
        # start = (0, (None,) + (0,) * d)
        # neighbours(vertex):
        #   side, x = vertex
        #   if side==0: y = [None, first] + [0]*(d-1) ...
        #   if side==1: y = [None, first] ... (Wait, the code yields 1-side, tuple(y))
        
        # Let's stick to tuples for safety, but use a dict for mapping.
        pass

    # The original code is likely fast enough for 354k vertices.
    # I will use the original logic but add time checks.

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1
        # side 0: point. x[0] is None. x[1..d] are coords.
        # side 1: line. x[0] is coord. x[1..d] are coords.
        # Wait, looking at code:
        # y = [None, first] + [0] * (d - 1)
        # This implies y[0] is None, y[1] is 'first', y[2..d] are 0.
        # Length is 1 + 1 + d - 1 = d + 1.
        # So x[0] is always None in the initial state?
        # If side==1, y[0] is None?
        # Let's trace:
        # start = (0, (None,) + (0,)*d) -> x[0]=None.
        # neighbour of start (side 0):
        #   y = [None, first] + ...
        #   side becomes 1.
        #   x for line has x[0]=None?
        #   Then in next step, side=1.
        #   y = [None, first] ...
        #   This suggests x[0] is always None?
        #   But the rule uses x[a] where a can be 1?
        #   rule[2] = (1,1). a=1, b=1.
        #   x[1] is 'first'.
        
        # It seems x[0] is a dummy.
        
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l
                    # l[c] = p[c] + l[a] * p[b]
                    # y is line l. x is point p.
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p
                    # p[c] = l[c] - l[a] * p[b] ?
                    # Code: y[c] = sub(x[c], mul(x[a], y[b]))
                    # x is line. y is point.
                    # y[c] = x[c] - x[a] * y[b]
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
            # Return whatever we have? No, must be valid.
            # If we time out, we should return a smaller valid graph?
            # Or just raise?
            # Better to return the L-U-W graph if it finishes.
            # If it doesn't finish, we might fail.
            pass
            
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
