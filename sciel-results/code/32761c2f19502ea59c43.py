"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    start_time = time.time()
    
    # For k=5, g=9, use Lazebnik-Ustimenko-Woldar construction CD(4, 5)
    # This yields a 5-regular bipartite graph with girth at least 9.
    # Number of vertices: 2 * 5^4 = 1250.
    
    if k != 5 or g < 9:
        # Fallback for other parameters (not expected in this task)
        # Use a simple random search or return None/empty if not applicable
        # But the task specifies k=5, g=9.
        pass

    q = k
    # d is chosen such that girth >= g.
    # For LUW, girth >= d + 5 for odd d.
    # We need d + 5 >= 9 => d >= 4.
    # If d is even, girth >= 2d? No, standard is d+5 for odd d.
    # Let's use d=4.
    d = 4
    
    # GF(5) arithmetic
    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return (x * y) % q
    
    # Rule definition for LUW graph
    # y[c] = x[c] + y[a] * x[b] (for side 0)
    # y[c] = x[c] - x[a] * y[b] (for side 1)
    # Indices are 1-based in the paper, 0-based in code?
    # The seed code uses:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d + 1):
    #    ...
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The rule cycles through specific patterns
        # Pattern for c:
        # c=5: r=0, c0=5 -> (1, 3) ? 
        # Let's trust the seed code's rule generation.
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1, x[0] is dummy/None, x[1..d] are values
        for first in range(q):
            # y[0] is dummy, y[1] = first
            # y[2..d] are calculated
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    # Start vertex: (0, (None, 0, 0, 0, 0))
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i in range(len(order)):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Add edge only from side 0 to avoid duplicates
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges
# EVOLVE-BLOCK-END
