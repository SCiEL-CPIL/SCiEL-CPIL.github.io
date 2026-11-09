<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
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
=======
"""Improved Lazebnik-Ustimenko-Woldar construction with deadline guard and precomputed tables."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    deadline = time.time() + 110  # safety margin before 120s limit
    
    q = k
    # For the LUW construction, we need d such that girth >= d + 5 (for odd d) or d + 6 (for even d)
    # We want the smallest d giving girth >= g
    # For odd d: girth >= d + 5, so need d >= g - 5, d odd
    # For even d: girth >= d + 6, so need d >= g - 6, d even
    # We pick the smaller total graph size
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4  # this is odd when g-5 is even, so g-4 is odd... let me reconsider
    
    # Actually: for LUW CD(d,q), the girth is at least d+5 if d is odd, d+6 if d is even.
    # We want girth >= g.
    # Option 1: d odd, d+5 >= g => d >= g-5, d odd
    # Option 2: d even, d+6 >= g => d >= g-6, d even
    # Pick the one with smaller d (smaller graph)
    d_odd = g - 5
    if d_odd % 2 == 0:
        d_odd += 1
    d_even = g - 6
    if d_even % 2 == 1:
        d_even += 1
    d = min(d_odd, d_even)
    d = max(3, d)
    
    # Precompute finite field operations
    if q == 4:
        # GF(4) with x^2 + x + 1 irreducible
        # Elements: 0, 1, a, a^2 where a^2 = a + 1
        # Represent as: 0->0, 1->1, a->2, a+1->3
        add_table = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
    else:
        # GF(p) for prime p = k
        add_table = [[(i + j) % q for j in range(q)] for i in range(q)]
        mul_table = [[(i * j) % q for j in range(q)] for i in range(q)]
    
    # Precompute the rule table: rule[c] = (a, b) for c = 2..d
    rule = [None] * (d + 1)
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    # Build the graph using BFS from the zero point
    # Vertices are (side, x) where side in {0,1} and x is a tuple of length d+1
    # x[0] is unused (None), x[1..d] are field elements
    # We represent x as a tuple for hashing
    
    start = (0, (0,) * (d + 1))  # all zeros, x[0]=0 unused
    index = {start: 0}
    order = [start]
    edges = []
    
    # Adjacency: for each vertex, compute its q neighbors
    # Point (side=0, p) is adjacent to line (side=1, l) if for all c=2..d: l[c] = p[c] + l[a]*p[b]
    # Line (side=1, l) is adjacent to point (side=0, p) if for all c=2..d: p[c] = l[c] - l[a]*p[b]
    
    # For a point (0, p): neighbors are lines l where l[1] is free, and l[c] = p[c] + l[a_c]*p[b_c] for c=2..d
    # For a line (1, l): neighbors are points p where p[1] is free, and p[c] = l[c] - l[a_c]*p[b_c] for c=2..d
    
    vertex_list = []
    for vertex in order:
        if time.time() > deadline:
            break
        side, x = vertex
        vid = index[vertex]
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = 0  # unused
            y[1] = first
            if side == 0:
                # Point to line: l[c] = p[c] + l[a]*p[b]
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
            else:
                # Line to point: p[c] = l[c] - l[a]*p[b]
                # p[c] = l[c] - l[a]*p[b] => need to solve for p[c]
                # But p[b] is already determined (b < c since rule has b < c typically)
                # Actually in the LUW construction, b is always less than c
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add_table[x[c]][add_table[0][mul_table[y[a]][x[b]]]]  # x[c] - y[a]*x[b]
                    # Wait, subtraction: x[c] - y[a]*x[b]
                    # In GF(p): (x[c] - y[a]*x[b]) % q
                    # In GF(4): add_table is also sub_table since char 2
            neighbor = (1 - side, tuple(y))
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            if side == 0:
                edges.append((vid, index[neighbor]))
    
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I need to reconsider the subtraction in GF(4). In characteristic 2, addition and subtraction are the same. Let me fix that and also verify the logic more carefully.

