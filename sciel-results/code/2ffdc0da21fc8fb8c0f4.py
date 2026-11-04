"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
    q = k
    # Choose minimal d such that girth >= g.
    # LUW graph CD(d,q) has girth d+5 if d odd, d+4 if d even (conservative).
    # We need the smallest d with guaranteed girth >= g.
    if g <= 5:
        d = 3
    else:
        # Try smallest d >= 3
        d = 3
        while True:
            girth_est = d + 5 if d % 2 == 1 else d + 4
            if girth_est >= g:
                break
            d += 1
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Precompute rule arrays for faster access
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]

    # Precompute multiplication table for GF(q)
    mul_table = [[mul(i, j) for j in range(q)] for i in range(q)]
    add_table = [[add(i, j) for j in range(q)] for i in range(q)]
    sub_table = [[sub(i, j) for j in range(q)] for i in range(q)]

    # Precompute all neighbors for each vertex type
    # For side 0 (points): neighbor is (1, y) where y[1]=first, y[c] = x[c] + y[a]*x[b]
    # For side 1 (lines): neighbor is (0, y) where y[1]=first, y[c] = x[c] - x[a]*y[b]
    
    def compute_neighbors_point(x):
        """Compute neighbors of a point vertex (side 0)."""
        result = []
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule_a[c], rule_b[c]
                y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
            result.append((1, tuple(y)))
        return result

    def compute_neighbors_line(x):
        """Compute neighbors of a line vertex (side 1)."""
        result = []
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule_a[c], rule_b[c]
                y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
            result.append((0, tuple(y)))
        return result

    start = (0, (0,) * (d + 1))
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if time.time() > deadline:
            break
        side, x = vertex
        if side == 0:
            neighbors = compute_neighbors_point(x)
        else:
            neighbors = compute_neighbors_line(x)
        for other in neighbors:
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
