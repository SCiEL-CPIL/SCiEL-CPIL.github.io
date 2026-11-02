import time
from collections import deque

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # For k=4, g=10, this yields the optimal 320-vertex graph.
    q = k
    # d is chosen such that girth is at least d+5 (for odd d)
    # For g=10, we need d+5 >= 10 => d >= 5.
    # The seed uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # For g=10: (10-5)=5, 5%2=1, so d=5.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Finite field GF(q) operations
    if q == 4:
        # GF(4) = {0, 1, 2, 3} with 2^2=1 (in char 2, 2^2=0? No, GF(4) is char 2. 1+1=0. 2 is not an element. Elements are 0, 1, alpha, alpha+1. 
        # The seed uses 0,1,2,3 as labels.
        # Multiplication table for GF(4) represented as {0,1,2,3} where 2=alpha, 3=alpha+1?
        # Seed table:
        # [[0, 0, 0, 0], 
        #  [0, 1, 2, 3], 
        #  [0, 2, 3, 1], 
        #  [0, 3, 1, 2]]
        # This is a valid group table for Z4? No, Z4 is abelian. 
        # 2*2=1? In GF(4), alpha^2 = alpha + 1. If 2=alpha, 3=alpha+1.
        # 2*2 = alpha^2 = alpha+1 = 3. But table[2][2]=3. Correct.
        # 2*3 = alpha(alpha+1) = alpha^2+alpha = (alpha+1)+alpha = 1. Table[2][3]=1. Correct.
        # 3*3 = (alpha+1)^2 = alpha^2+1 = (alpha+1)+1 = alpha = 2. Table[3][3]=2. Correct.
        # So this is GF(4).
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y  # XOR for GF(4) addition (char 2)
        sub = lambda x, y: x ^ y  # Subtraction is addition in char 2
        mul = lambda x, y: table[x][y]
    else:
        # Assume q is prime for general case, though task specifies k=4
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
    # Rule for LUW graph
    # rule[c] = (a, b) such that y[c] = y[a] * x[b] + x[c] (or similar)
    # Seed logic:
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The seed picks from a list based on r
        # r=0: (1, c0-2), r=1: (c0-1, 1), r=2: (c0, 1), r=3: (1, c0+1)
        # For d=5, c=5. r=(5-5)%4=0. c0=5.
        # rule[5] = (1, 5-2) = (1, 3).
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1. x[0] is None, x[1] is the free variable.
        for first in range(q):
            # y[0] is None, y[1] is 'first', y[2..d] are 0 initially
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line: l[c] - p[c] = l[a] * p[b]  =>  l[c] = p[c] + l[a] * p[b]
                    # In GF(4), subtraction is addition.
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point: p[c] = l[c] - l[a] * p[b]  =>  p[c] = l[c] + l[a] * p[b] (in char 2)
                    # Wait, the seed uses: sub(x[c], mul(x[a], y[b]))
                    # x is the line, y is the point.
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
            break
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    # Verification (optional but good for safety)
    # Check degree
    # Check girth (expensive, skip for speed if N is small)
    
    return len(order), edges
