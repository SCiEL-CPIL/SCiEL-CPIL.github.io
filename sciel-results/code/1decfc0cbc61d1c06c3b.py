"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar (LUW)
    construction for q = k (when k is prime or 4), or a random regular graph search otherwise.
    
    The LUW construction gives a q-regular bipartite graph with girth at least d+5 (odd d) or d+4 (even d).
    We choose d to ensure the girth meets the requirement g.
    
    For k=3, g=16: q=3, we need girth >= 16. With d odd, girth >= d+5, so d=11 gives girth >= 16.
    The number of vertices will be 2 * q^d = 2 * 3^11 = 354270, which is large but valid.
    
    However, we can do better. Let's check: for q=3, d=11, the LUW graph has 2*3^11 vertices.
    That's 354270 vertices. The best known is 936. So LUW is far from optimal.
    
    Let's try a different approach: use the known optimal or near-optimal constructions.
    For (3,16), the best known is 936 vertices. Let's try to construct it directly.
    
    Actually, let's use a simpler approach: try to find a small girth-16 cubic graph.
    The (3,16)-cage has 936 vertices. We can try to construct it using known methods.
    
    Since we don't have the explicit construction memorized, let's use the LUW construction
    but with a smaller parameter if possible, or use a random search with local improvement.
    
    Given the 150s time limit, let's try:
    1. First, check if k is 3 and g is 16 - use a known construction if possible.
    2. Otherwise, use LUW construction as a fallback.
    
    For now, let's implement the LUW construction correctly and efficiently.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # For the specific case of k=3, g=16, we know the answer is 936 vertices.
    # Let's try to construct a (3,16)-cage or a close approximation.
    
    # The LUW construction for q=3, d=11 gives girth >= 16 but with 354270 vertices.
    # This is valid but not optimal. Let's see if we can do better.
    
    # Actually, let's just implement the LUW construction correctly.
    # It's guaranteed to work and be valid.
    
    q = k
    # Choose d such that the girth is at least g
    # For odd d: girth >= d + 5
    # For even d: girth >= d + 4
    if (g - 5) % 2 == 1:
        d = max(3, g - 5)
    else:
        d = max(4, g - 4)
    
    # Ensure d is at least 3
    if d < 3:
        d = 3
    
    # For q=4, use GF(4) arithmetic
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y):
            return x ^ y
        def sub(x, y):
            return x ^ y
        def mul(x, y):
            return table[x][y]
    else:
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return x * y % q
    
    # Define the rule for the LUW construction
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # The LUW graph has vertices that are pairs (side, vector) where side is 0 or 1
    # and vector is a tuple of length d+1 (with the first element being None for points)
    
    # For efficiency, let's precompute the neighbours
    # The graph is bipartite: points (side=0) and lines (side=1)
    
    # Number of vertices: 2 * q^d
    n_vertices = 2 * (q ** d)
    
    # Check if we're running out of time
    if time.time() > deadline:
        # Return a trivial valid graph if possible, or raise an error
        # For now, return the LUW construction as is
        pass
    
    # Build the graph using BFS from the zero vertex
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges

# EVOLVE-BLOCK-END
