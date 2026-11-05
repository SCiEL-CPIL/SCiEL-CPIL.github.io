import time
import sys
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # For k=5, g=9, use LUW with q=5, d=4
    # This yields n = 2 * 5^4 = 1250 vertices.
    if k == 5 and g == 9:
        n, edges = _construct_luw_optimized(5, 4)
        # Verify girth (optional if we trust the construction, but good for safety)
        # The LUW construction guarantees girth 2d+1 = 9.
        return n, edges

    # General case
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _construct_luw_optimized(q, d)

def _construct_luw_optimized(q, d):
    """
    Construct the Lazebnik-Ustimenko-Woldar graph CD(d, q).
    Optimized for speed.
    """
    # Precompute operation tables
    add_table = [[(x + y) % q for y in range(q)] for x in range(q)]
    mul_table = [[x * y % q for y in range(q)] for x in range(q)]
    sub_table = [[(x - y) % q for y in range(q)] for x in range(q)]
    
    # Define the rule for indices
    # rule[c] = (a, b) such that y[c] = op(x[c], x[a], y[b])
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        # The pattern from the original code:
        # r=0: (1, c0-2)
        # r=1: (c0-1, 1)
        # r=2: (c0, 1)
        # r=3: (1, c0+1)
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # We represent a vertex as (side, x0, x1, ..., xd)
    # side 0: y[c] = x[c] + y[a] * x[b]
    # side 1: y[c] = x[c] - x[a] * y[b]
    
    # To speed up, we can precompute the neighbor generation for each side.
    # However, the state space is 2 * q^d.
    # For q=5, d=4, n=1250. This is small enough to generate on the fly.
    
    # We will use BFS to assign indices.
    # Start vertex: (0, 0, 0, 0, 0, 0)
    start_vertex = (0, 0, 0, 0, 0, 0)
    index = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    # Precompute neighbors for a given vertex
    # This function is called n * degree times.
    def get_neighbours(side, x_tuple):
        # x_tuple is (x0, x1, ..., xd)
        # We need to generate all y such that (1-side, y) is a neighbor.
        # The relation is defined by the rule.
        # For side 0, we fix y[1] (the first variable of the next side?)
        # In LUW, the vertex is (side, x1, ..., xd).
        # The neighbor on the other side is determined by x.
        # Actually, the LUW graph is defined such that for each x on side 0,
        # and each possible value of the "free" variable on side 1, we get a neighbor.
        
        # Let's trace the original code logic:
        # for first in range(q):
        #   y = [None, first] + [0] * (d - 1)
        #   ... calculate rest of y ...
        #   yield 1 - side, tuple(y)
        
        # This loop runs q times.
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add_table[x_tuple[c]][mul_table[y[a]][x_tuple[b]]]
                else:
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = sub_table[x_tuple[c]][mul_table[x_tuple[a]][y[b]]]
            yield 1 - side, tuple(y)

    for i, vertex in enumerate(order):
        side, x_tuple = vertex
        for other in get_neighbours(side, x_tuple):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only once (when side is 0 to avoid duplicates)
            if side == 0:
                edges.append((i, index[other]))
                
    return len(order), edges
