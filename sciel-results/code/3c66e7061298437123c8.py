import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # LUW Construction
    q = k
    # d must be such that girth >= g
    # Girth of CD(d, q) is >= d + 5
    # So d + 5 >= g => d >= g - 5
    # Also d must be >= 3
    d = max(3, g - 5)
    
    # For q=4, use GF(4)
    if q == 4:
        # GF(4) = {0, 1, 2, 3} with x^2+x+1=0
        # 1^2=1, 2^2=3, 3^2=2
        # 1+1=0, 1+2=3, 1+3=2
        # 2+2=0, 2+3=1, 3+3=0
        # 1*1=1, 1*2=2, 1*3=3
        # 2*2=3, 2*3=1, 3*3=2
        add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return add_table[x][y]
        def sub(x, y): return add_table[x][y] # char 2
        def mul(x, y): return mul_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Rule for LUW
    # rule[c] = (a, b) such that y[c] = y[a] * x[b] + x[c] (or similar)
    # Standard LUW rule:
    # For c=2: y[2] = x[2] + y[1]*x[1] ? No.
    # The seed code has:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d+1):
    #    r, c0 = (c-5)%4, c - (c-5)%4
    #    rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None/side marker?
        # In seed: x = (None,) + (0,)*d
        # So x[0] is None, x[1..d] are values.
        # Wait, seed: y = [None, first] + [0]*(d-1)
        # So y[0] is None, y[1] is 'first', y[2..d] are computed.
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point to Line
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line to Point
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to find connected component
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

# EVOLVE-BLOCK-START
def construct(k, g):
    # Use the refined LUW construction
    return _construct_luw(k, g)

def _construct_luw(k, g):
    import time
    start_time = time.time()
    deadline = start_time + 110
    
    q = k
    d = max(3, g - 5)
    
    if q == 4:
        add_table = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def add(x, y): return add_table[x][y]
        def sub(x, y): return add_table[x][y]
        def mul(x, y): return mul_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

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
    
    for i in range(len(order)):
        if time.time() > deadline:
            break
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
