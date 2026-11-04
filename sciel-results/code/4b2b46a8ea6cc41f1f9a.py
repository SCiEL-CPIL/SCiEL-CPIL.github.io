"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
def construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    
    Returns the connected component of the zero point.
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    
    q = k
    # For odd d, girth >= d+5. We need d+5 >= g, so d >= g-5.
    # d must be odd for the standard LUW construction.
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # ensure odd
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Precompute rule table
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        if c == 2:
            rule_a[c], rule_b[c] = 1, 1
        elif c == 3:
            rule_a[c], rule_b[c] = 2, 1
        elif c == 4:
            rule_a[c], rule_b[c] = 1, 2
        else:
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule_a[c], rule_b[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            
    def neighbours_fast(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is None for side 0, or 0 for side 1?
        # In the original code: start = (0, (None,) + (0,) * d)
        # So x[0] is None for side 0.
        # For side 1, the vertex is (1, tuple(y)) where y[0] is the 'first' value.
        # Wait, the original code yields (1 - side, tuple(y)).
        # If side=0, yields (1, tuple(y)). y[0] = first.
        # If side=1, yields (0, tuple(y)). y[0] = first.
        # But for side=1, the input x[0] is used?
        # Let's trace:
        # vertex = (side, x).
        # if side == 0: x[0] is None.
        #   y[0] = first.
        #   y[c] = x[c] + y[a]*x[b].
        #   Here x[b] is accessed. If b=0, x[0] is None.
        #   In the rule, b is at least 1?
        #   c=2: a=1, b=1. x[1] is valid.
        #   c=3: a=2, b=1. x[1] is valid.
        #   c=4: a=1, b=2. x[2] is valid.
        #   c>=5: a, b are in range 1..d.
        #   So x[0] is never accessed when side=0.
        # if side == 1: x[0] is 'first' (an integer).
        #   y[0] = first.
        #   y[c] = x[c] - x[a]*y[b].
        #   Here y[b] is accessed.
        #   If b=0, y[0] is first.
        #   So x[0] is not accessed directly, but y[0] is.
        #   Wait, the formula is y[c] = x[c] - x[a] * y[b].
        #   x[a] is accessed. a is in 1..d.
        #   y[b] is accessed. b is in 0..d.
        #   So x[0] is never accessed.
        #   However, x is a tuple of length d+1.
        #   For side=1, x[0] is an integer.
        #   For side=0, x[0] is None.
        
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = first
            if side == 0:
                # point to line
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = (x[c] + mul(y[a], x[b])) % q
            else:
                # line to point
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = (x[c] - mul(x[a], y[b])) % q
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        vertex = order[i]
        i += 1
        v_idx = index[vertex]
        for other in neighbours_fast(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((v_idx, index[other]))
        
        if time.time() > deadline:
            break
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
