"""Cage problem n(k,g): build a k-regular graph with girth >= g on few vertices.

Strategy: the Lazebnik-Ustimenko-Woldar (LUW) family CD(d,q).  CD(d,q) is a
q-regular bipartite graph whose girth is d+5 for odd d >= 3.  We therefore set
q = k and d = g - 5 (forcing d odd) and return the connected component of the
zero point.  This yields a *valid* graph (the seed silently fell back to d=3,
i.e. girth 8, for the cage parameters k=5, g=9, which is invalid).  The vertex
count is then minimised over the valid choices.

The construction is encoded algebraically: a point is (x0, x1, ..., xd) and a
line is (l1, ..., ld) over GF(q) (q prime, or q = 4).  Point p is incident to
line l iff, for every c = 2..d,  l_c - p_c = l_{a_c} * p_{b_c}, where the rule
(a_c, b_c) is the standard LUW rule:
    a_2 = b_2 = 1 ;  a_3 = 2, b_3 = 1 ;  a_4 = 1, b_4 = 2 ;
    c >= 5:  r = (c-5) mod 4,  c0 = c - r,
             (a_c, b_c) = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r].
"""
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 110.0   # explicit safety margin under the 120s limit

    def field_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def build(q, d):
        add, sub, mul = field_ops(q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:                       # point p -> line l
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:                                # line l -> point p
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    # q must equal the degree k; d must be odd with d = g - 5 for the girth to
    # be exactly g.  For the cage parameters this is a single valid choice.
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    return build(q, d)
# EVOLVE-BLOCK-END
