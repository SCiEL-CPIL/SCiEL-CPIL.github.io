"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # Girth guarantee: d+5 if d odd, d+4 if d even.
    # Choose minimal d such that guaranteed girth >= g.
    q = k

    # Find minimal d >= 3 with girth guarantee >= g
    d = 3
    while True:
        girth_est = d + 5 if d % 2 == 1 else d + 4
        if girth_est >= g:
            break
        d += 1

    # GF(q) operations - precompute tables for speed
    if q == 4:
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add_table = [[i ^ j for j in range(4)] for i in range(4)]
        sub_table = [[i ^ j for j in range(4)] for i in range(4)]  # same as add in GF(4)
    else:
        mul_table = [[(x * y) % q for y in range(q)] for x in range(q)]
        add_table = [[(x + y) % q for y in range(q)] for x in range(q)]
        sub_table = [[(x - y) % q for y in range(q)] for x in range(q)]

    # Precompute rules for the LUW construction
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            a, b = 1, c0 - 2
        elif r == 1:
            a, b = c0 - 1, 1
        elif r == 2:
            a, b = c0, 1
        else:
            a, b = 1, c0 + 1
        rule_a[c] = a
        rule_b[c] = b

    # Precompute neighbor transformations for each side
    # For side 0: y[c] = x[c] + y[rule_a[c]] * x[rule_b[c]]
    # For side 1: y[c] = x[c] - x[rule_a[c]] * y[rule_b[c]]
    # We precompute the "transformation" for each possible first coordinate

    # For side 0, given x (tuple of length d+1) and first coord f:
    # y[0] = 0, y[1] = f, and for c >= 2: y[c] = add(x[c], mul(y[rule_a[c]], x[rule_b[c]]))
    # This is sequential since y[rule_a[c]] may depend on earlier y values

    # For speed, we'll use a cached approach: precompute for each (side, first_coord)
    # the transformation as a function of x

    # Actually, let's just optimize the inner loop by avoiding function call overhead
    # and using local variables

    def compute_neighbours_fast(side, x):
        """Compute all neighbors of vertex (side, x)."""
        result = []
        if side == 0:
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a = rule_a[c]
                    b = rule_b[c]
                    y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                result.append((1, tuple(y)))
        else:
            for first in range(q):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a = rule_a[c]
                    b = rule_b[c]
                    y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
                result.append((0, tuple(y)))
        return result

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []

    # BFS to build the connected component
    head = 0

    while head < len(order):
        if time.time() > deadline:
            break
        vertex = order[head]
        head += 1
        side, x = vertex
        vid = index[vertex]

        for other in compute_neighbours_fast(side, x):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge if from Point side (0) to avoid duplicates
            if side == 0:
                edges.append((vid, index[other]))

    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
