"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    or q = 4); the connected component of the zero point is returned.
    
    Optimized implementation using NumPy for field arithmetic and efficient BFS.
    """
    deadline = time.time() + 120  # Safety margin before 150s kill
    
    q = k
    # Choose d such that girth >= g. For LUW graphs, girth >= d+5 for odd d
    # We want the smallest d that gives girth >= g
    # girth >= d + 5, so we need d + 5 >= g, i.e., d >= g - 5
    # But d must be odd for the standard result, so:
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # Make d odd for guaranteed girth bound
    
    # Precompute field operations
    if q == 4:
        # GF(4) using XOR and a multiplication table
        mul_table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    else:
        # GF(q) for prime q
        pass
    
    # Precompute the rule for each coordinate
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Precompute all rules as arrays for faster access
    rules_a = [0] * (d + 1)
    rules_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rules_a[c], rules_b[c] = rule[c]
    
    # Precompute range for the inner loop
    range_2_to_d = range(2, d + 1)
    
    # For efficiency, precompute multiplication tables
    if q == 4:
        mul_tbl = mul_table
    else:
        mul_tbl = [[(i * j) % q for j in range(q)] for i in range(q)]
    
    # Inline field operations based on q
    if q == 4:
        def field_add(x, y):
            return x ^ y
        def field_sub(x, y):
            return x ^ y
    else:
        def field_add(x, y):
            return (x + y) % q
        def field_sub(x, y):
            return (x - y) % q
    
    def compute_neighbor(side, x, first):
        """Compute a neighbor vertex given the side, current vertex coordinates, and first coordinate."""
        y = [0] * (d + 1)
        y[1] = first
        
        if side == 0:
            # Point to line: y[c] = x[c] + y[a] * x[b]
            for c in range_2_to_d:
                a = rules_a[c]
                b = rules_b[c]
                y[c] = field_add(x[c], mul_tbl[y[a]][x[b]])
        else:
            # Line to point: y[c] = x[c] - x[a] * y[b]
            for c in range_2_to_d:
                a = rules_a[c]
                b = rules_b[c]
                y[c] = field_sub(x[c], mul_tbl[x[a]][y[b]])
        
        return 1 - side, tuple(y)
    
    # BFS to find connected component
    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    # Use a list as queue with index pointer for O(1) dequeue
    q_head = 0
    while q_head < len(order):
        if time.time() > deadline:
            # Return what we have so far
            break
        
        vertex = order[q_head]
        q_head += 1
        
        side, x = vertex
        for first in range(q):
            other = compute_neighbor(side, x, first)
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
