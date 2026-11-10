"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    deadline = time.time() + 140
    
    q = k
    # For girth g, need girth(CD(d, q)) >= g.
    # Girth is d+5 for odd d, d+4 for even d.
    # We prefer odd d for better girth per vertex count?
    # Actually, we just need the smallest d such that girth >= g.
    
    # Find minimal d
    d = 1
    while True:
        if d % 2 == 1:
            girth = d + 5
        else:
            girth = d + 4
        if girth >= g:
            break
        d += 1
        
    # If d is even, we might be able to use d-1 if it's odd and girth(d-1) >= g?
    # No, we want minimal d to minimize vertices 2*q^d.
    # But if d is even, girth is lower.
    # Example: g=9.
    # d=4 (even) -> girth 8 < 9.
    # d=5 (odd) -> girth 10 >= 9.
    # So d=5 is minimal.
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Precompute rule
    rule = [None, None, (1, 1), (2, 1), (1, 2)]
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule.append([(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r])
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, x = vertex
        for first in range(q):
            y_list = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y_list[c] = add(x[c], mul(y_list[a], x[b]))
                else:
                    y_list[c] = sub(x[c], mul(x[a], y_list[b]))
            other = (1 - side, tuple(y_list))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
