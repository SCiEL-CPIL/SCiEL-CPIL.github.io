import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    # We want to minimize n = 2 * q^d subject to girth >= g.
    # Girth of CD(d, q) is at least 2d+2 (even d) or 2d+1 (odd d) ?
    # Let's assume girth is approx 2d+2.
    # We need 2d+2 >= g => d >= (g-2)/2.
    # For g=9, d >= 3.5 => d=4.
    # For g=10, d >= 4.
    # For g=11, d >= 4.5 => d=5.
    
    q = k
    # Determine d
    # If g is odd, we need girth >= g.
    # If girth(d) = 2d+2 for even d, then for g=9, d=4 gives 10.
    # If girth(d) = 2d+1 for odd d, then for g=9, d=4 is better than d=3 (girth 7? or 8?).
    # Let's just iterate d from 1 upwards and check a lower bound on girth.
    # A safe lower bound for girth of CD(d,q) is 2d+2.
    # So we need 2d+2 >= g.
    d = 1
    while 2 * d + 2 < g:
        d += 1
        
    # For q=4, use GF(4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
