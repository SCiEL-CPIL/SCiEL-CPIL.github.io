"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 115.0  # Safety margin

    # Hardcoded known optimal or near-optimal cages for small parameters
    def petersen_graph():
        # 3-regular, girth 5, 10 vertices
        edges = []
        for i in range(5):
            edges.append((i, (i+1) % 5))
            edges.append((i, 5+i))
            edges.append((5+i, 5+(i+2) % 5))
        return 10, edges

    def heawood_graph():
        # 3-regular, girth 6, 14 vertices
        edges = []
        lines = [
            (0, 1, 2), (0, 3, 4), (0, 5, 6),
            (1, 3, 5), (1, 4, 6), (2, 3, 6),
            (2, 4, 5)
        ]
        for i, line in enumerate(lines):
            for p in line:
                edges.append((p, 7+i))
        return 14, edges

    # Lookup table
    known_cages = {
        (3, 3): (3, [(0,1), (1,2), (0,2)]),
        (3, 4): (6, [(0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)]),
        (3, 5): petersen_graph(),
        (3, 6): heawood_graph(),
        (4, 3): (4, [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]),
        (4, 4): (8, [(i, j) for i in range(4) for j in range(4, 8)]),
        (5, 3): (5, [(i, j) for i in range(5) for j in range(i+1, 5)]),
        (5, 4): (10, [(i, j) for i in range(5) for j in range(5, 10)]),
    }

    if (k, g) in known_cages:
        return known_cages[(k, g)]

    # Use the modular tensor product construction
    # This builds a k-regular graph with controlled girth
    # Strategy: use a covering graph construction over Z_p x Z_m
    
    # For k=5, g=9: we want a 5-regular graph with girth >= 9
    # Using the modular construction:
    # Vertices: (x, y) where x in Z_p, y in Z_m
    # Edges defined by: for each vertex (x, y), connect to (x + a_i, y + b_i) for i=1..k
    # where a_i, b_i are chosen to avoid short cycles
    
    # A simpler approach: use a circulant-like construction
    # For k=5, we can use a 5-regular circulant graph on n vertices
    # where the connection set is {d1, d2, d3, d4, d5}
    
    # For girth 9, we need to carefully choose the connection set
    # Let's use a known construction: the (5,9)-cage or near-cage
    
    # Since we don't have the exact 1116-vertex cage, we'll use a construction
    # that guarantees girth >= g
    
    # Use the Lazebnik-Ustimenko-Wold construction adapted for this case
    # For k=5, g=9: d = g - 5 = 4, but d must be odd, so d = 5
    # This gives girth >= d + 5 = 10 >= 9
    
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)
    
    q = k
    
    # Precompute field tables
    if q == 4:
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add_table = [[i ^ j for j in range(4)] for i in range(4)]
        sub_table = add_table
    else:
        mul_table = [[(i * j) % q for j in range(q)] for i in range(q)]
        add_table = [[(i + j) % q for j in range(q)] for i in range(q)]
        sub_table = [[(i - j) % q for j in range(q)] for i in range(q)]

    # Rules for c = 2..d
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        a, b = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        rule_a[c] = a
        rule_b[c] = b

    # Precompute powers of q
    powers = [1] * (d + 1)
    for i in range(1, d + 1):
        powers[i] = powers[i-1] * q

    start_coeff = 0
    start_vertex = (0, start_coeff)
    
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    queue = deque([0])
    
    while queue:
        if time.time() > deadline - 5:
            break
        
        vi = queue.popleft()
        side, coeffs_int = order[vi]
        
        # Decode coefficients
        x = [0] * (d + 1)
        temp = coeffs_int
        for i in range(d, 0, -1):
            x[i] = temp % q
            temp //= q
            
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a = rule_a[c]
                b = rule_b[c]
                if side == 0:
                    y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                else:
                    y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]
            
            # Encode y into integer
            y_int = 0
            for i in range(1, d + 1):
                y_int = y_int * q + y[i]
                
            other = (1 - side, y_int)
            
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            
            if side == 0:
                edges.append((vi, index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
