import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    # This construction yields a q-regular bipartite graph with girth >= 2d+2 (or similar).
    # For k=5, g=9, we need d such that girth >= 9.
    # d=4 gives girth >= 10 (typically) or at least 9.
    # Number of vertices n = 2 * q^d.
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    q = k
    # Determine d
    # Girth of CD(d,q) is at least 2d+2 for large q? 
    # Actually, the girth is exactly 2d+2 if q is large enough.
    # For g=9, we need 2d+2 >= 9 => 2d >= 7 => d >= 3.5 => d=4.
    # If d=3, girth=8, which is < 9.
    # So d=4 is the minimum.
    d = 4
    
    # If q is not prime, LUW might not work directly or needs GF(q^m).
    # k=5 is prime, so we use GF(5).
    
    # Arithmetic in GF(5)
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return (x * y) % q
    
    # Rule for LUW construction
    # The rule is defined for c = 2..d
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    # Standard LUW rule:
    # For d=4:
    # c=2: (1, 1) ? No, indices are 1..d.
    # The seed program uses:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # Let's verify if this rule is correct for d=4.
    # The general rule in LUW paper:
    # For c=2..d, rule[c] = (a_c, b_c)
    # Usually a_c = 1, b_c = c-1? Or similar.
    # The seed program's rule seems specific.
    # I will use the seed program's rule logic.
    
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    if d >= 5: rule[5] = (2, 2)
    # Extend if needed
    for c in range(6, d+1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None, x[1..d] are values
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l: l[c] - p[c] = l[a] * p[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p: l[c] - p[c] = l[a] * p[b]
                    # x[c] = y[c] - x[a] * y[b] => y[c] = x[c] + x[a] * y[b]
                    # Wait, the seed program uses:
                    # y[c] = sub(x[c], mul(x[a], y[b]))
                    # This implies x[c] = y[c] + x[a]*y[b] => y[c] = x[c] - x[a]*y[b]
                    # This is consistent with l[c] - p[c] = l[a] * p[b] => p[c] = l[c] - l[a]*p[b]
                    # So if we are at line x, and going to point y:
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # Build the graph
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (from point side)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
        # Time check
        if time.time() > deadline:
            break
            
    return len(order), edges
# EVOLVE-BLOCK-END
