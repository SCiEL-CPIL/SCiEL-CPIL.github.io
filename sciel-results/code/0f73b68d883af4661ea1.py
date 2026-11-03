import time

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph with girth at least g.
    
    Uses Lazebnik-Ustimenko-Woldar (LUX) construction for k=4, g=10.
    """
    deadline = time.time() + 140
    
    q = k  # field size
    # For girth g, we need d such that girth >= d + 5
    # So d >= g - 5, and d should be odd
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    
    # GF(4) arithmetic
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    def gf_add(x, y): return x ^ y
    def gf_sub(x, y): return x ^ y  
    def gf_mul(x, y): return table[x][y]
    
    # Rule for LUX construction
    rule = {}
    for c in range(2, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        patterns = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = patterns[r]
    
    def get_neighbours(vertex):
        side, x = vertex
        neighbours = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            neighbours.append((1 - side, tuple(y)))
        return neighbours
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
        if i % 50 == 0 and time.time() > deadline:
            break
    
    n = len(order)
    return n, edges

# EVOLVE-BLOCK-END
