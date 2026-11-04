"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    # We must choose the smallest d >= 3 such that the guaranteed girth is at least g.
    # Girth guarantee: d+5 if d is odd, d+4 if d is even.
    
    # Calculate minimal d
    # For odd d: d >= g - 5
    # For even d: d >= g - 4
    # We check the smallest candidates.
    
    min_d_odd = max(3, g - 5)
    if min_d_odd % 2 == 0:
        min_d_odd += 1
        
    min_d_even = max(4, g - 4)
    if min_d_even % 2 == 1:
        min_d_even += 1
        
    d = min(min_d_odd, min_d_even)
    
    q = k
    
    # GF(q) operations
    if q == 4:
        # GF(4) using XOR for add/sub and a multiplication table
        # 0*0=0, 0*1=0, 0*2=0, 0*3=0
        # 1*0=0, 1*1=1, 1*2=2, 1*3=3
        # 2*0=0, 2*1=2, 2*2=3, 2*3=1
        # 3*0=0, 3*1=3, 3*2=1, 3*3=2
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return mul_table[x][y]
    else:
        # Assume q is prime for standard GF(q)
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Precompute rules for the LUW construction
    # rule[c] = (a, b) such that y[c] = x[c] + y[a]*x[b] (for points)
    # and y[c] = x[c] - x[a]*y[b] (for lines)
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    
    # Base cases
    rule_a[2], rule_b[2] = 1, 1
    rule_a[3], rule_b[3] = 2, 1
    rule_a[4], rule_b[4] = 1, 2
    
    for c in range(5, d + 1):
        # Pattern repeats every 4
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            a, b = 1, c0 - 2
        elif r == 1:
            a, b = c0 - 1, 1
        elif r == 2:
            a, b = c0, 1
        else: # r == 3
            a, b = 1, c0 + 1
        rule_a[c] = a
        rule_b[c] = b

    def compute_neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1
        result = []
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            if side == 0:
                # Point to Line: y[c] = x[c] + y[a] * x[b]
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                # Line to Point: y[c] = x[c] - x[a] * y[b]
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            result.append((1 - side, tuple(y)))
        return result

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS to build the connected component
    # The graph is regular, so if we visit all reachable nodes, we have the component.
    # For LUW graphs, the zero vertex is in the largest component.
    
    for vertex in order:
        if time.time() > deadline:
            break
            
        for other in compute_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            
            # Add edge if from Point side (0) to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
# EVOLVE-BLOCK-END
