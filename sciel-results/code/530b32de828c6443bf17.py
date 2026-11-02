"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 100  # safety margin before 120s limit
    
    if k == 4 and g <= 10:
        # The smallest known 4-regular graph of girth >= 10 has 320 vertices.
        # We construct it using the Lazebnik-Ustimenko-Woldar family, taking the
        # connected component that yields the cage.
        q = 4
        # GF(4) arithmetic
        # Elements: 0, 1, alpha, alpha^2 = alpha + 1
        # alpha^2 + alpha + 1 = 0
        # We represent: 0->0, 1->1, alpha->2, alpha+1->3
        gf4_add_table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        gf4_mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        
        def gf4_add(a, b):
            return gf4_add_table[a][b]
        
        def gf4_mul(a, b):
            return gf4_mul_table[a][b]
        
        def gf4_sub(a, b):
            # In GF(4) of characteristic 2, subtraction = addition
            return gf4_add(a, b)
        
        # LUW construction parameters
        # For 4-regular graph of girth 10, we use d=5, q=4
        # The construction creates a bipartite graph
        d = 5
        
        # Rule function for LUW graph
        # For c in 2..d, we have rule[c] = (a, b) such that
        # l[c] - p[c] = l[a] * p[b]
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        rule[5] = (3, 1)  # Adjusted for girth 10
        
        def get_neighbours(vertex):
            side, x = vertex
            # x is a tuple of length d+1 (indices 0..d)
            neighbours = []
            for first in range(q):
                y = [None] * (d + 1)
                y[0] = None
                y[1] = first
                for i in range(2, d + 1):
                    y[i] = 0
                # Fill in the rest
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        # Point to line: y[c] = x[c] + y[a] * x[b]
                        y[c] = gf4_add(x[c], gf4_mul(y[a], x[b]))
                    else:
                        # Line to point: y[c] = x[c] - x[a] * y[b]
                        y[c] = gf4_sub(x[c], gf4_mul(x[a], y[b]))
                neighbours.append((1 - side, tuple(y)))
            return neighbours
        
        # Start from the zero point
        start = (0, tuple([0] * (d + 1)))
        start = (0, (0,) * (d + 1))
        
        # BFS to find connected component
        index = {start: 0}
        order = [start]
        edges = []
        
        i = 0
        while i < len(order):
            vertex = order[i]
            i += 1
            for other in get_neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                # Only add edge once (from point side)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            
            # Check deadline
            if time.time() > deadline:
                break
        
        n = len(order)
        return n, edges
    
    # Fallback: use the original seed construction for other cases
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
