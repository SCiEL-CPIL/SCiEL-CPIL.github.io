"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using a de Bruijn-type
    finite-state machine construction.
    
    We use the standard de Bruijn graph B(k, m) which is (k+1)-regular directed,
    but we adapt it to create a k-regular undirected graph with large girth.
    
    Approach: Use a circulant-based or algebraic construction that guarantees
    girth >= g. For the specific case of 3-regular girth 16, we need a clever
    construction.
    
    A practical approach: use the known explicit constructions from the cage
    problem literature. For small k and g, we can use:
    1. The (3,5) cage: Petersen graph (10 vertices)
    2. The (3,6) cage: Heawood graph (14 vertices)
    3. The (3,7) cage: Foster graph (90 vertices)
    4. For larger girth, use algebraic constructions or search.
    
    Given the 120s time limit, for the general case we'll use a combination of:
    - Known small cages when applicable
    - A BFS-based verification and construction for the general case
    
    For the de Bruijn approach: We construct a k-regular graph by taking
    vertices as words of length m over alphabet {0,...,k-1}, and connecting
    based on shift operations. The key insight is that a de Bruijn graph B(k,m)
    has girth related to m, and we can make it undirected and k-regular by
    appropriate quotienting.
    
    Actually, let me use a more direct approach: for k=3, g=16, the smallest
    known is 936 vertices. We'll use a construction based on the (3,16) cage
    or a close approximation.
    
    For a general solution within time limits, I'll use:
    1. If (k,g) is a known small cage, return it
    2. Otherwise, use a greedy/structured construction with verification
    """
    import sys
    sys.setrecursionlimit(100000)
    
    # Known small cages
    known_cages = {
        (3, 5): (10, [(0,1),(0,4),(0,5),(1,2),(1,5),(1,9),(2,3),(2,8),(2,9),(3,4),(3,7),(3,8),(4,6),(4,7),(5,6),(5,7),(6,8),(6,9),(7,8),(7,9)]),
        (3, 6): (14, [(0,1),(0,5),(0,10),(1,2),(1,6),(1,11),(2,3),(2,7),(2,12),(3,4),(3,8),(3,13),(4,5),(4,9),(4,13),(5,6),(5,10),(5,11),(6,7),(6,11),(6,12),(7,8),(7,12),(7,13),(8,9),(8,13),(8,14),(9,10),(9,14),(10,11),(10,14),(11,12),(12,13),(13,14)]),
    }
    
    if (k, g) in known_cages:
        return known_cages[(k, g)]
    
    # For the general case, use a structured construction.
    # The de Bruijn graph B(k, m) has k^m vertices and is (k+1)-regular directed.
    # To get a k-regular undirected graph, we can use a different approach.
    
    # Alternative: Use a circulant graph or algebraic construction.
    # For k=3, we can try a (3, g) cage construction.
    
    # Practical approach: Use a BFS to build a k-regular graph with large girth
    # by ensuring no short cycles. This is the "greedy" approach.
    
    # Given time constraints, let's use a simple but valid construction:
    # The complete bipartite graph K_{k,k} has girth 4, which is too small.
    
    # Better: Use a random k-regular graph and verify girth.
    # But random may not give girth >= g for large g.
    
    # For the specific task (3,16), let's use a known construction or
    # a fallback that at least returns a valid k-regular graph.
    
    # Fallback: Use a cycle of length n where n is large enough, but cycles
    # are 2-regular. For k=3, we need a 3-regular graph.
    
    # Let's use a lollipop or barbell construction, or better, a known 3-regular
    # graph with large girth.
    
    # The (3,16) cage has 936 vertices. Let's construct a 3-regular graph
    # with 936 vertices and hope it has girth >= 16, or use a construction
    # that guarantees it.
    
    # Actually, the safest approach for a general (k,g) within time:
    # Use the Lazebnik-Ustimenko-Woldar construction as in the seed, but
    # optimize the parameters.
    
    # Let me reimplement the LUK construction more carefully.
    
    q = k
    # d is the diameter-related parameter; girth >= d + 5 for odd d
    # We need girth >= g, so d + 5 >= g => d >= g - 5
    # For odd d: d = g - 5 if (g-5) is odd, else g - 4
    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    # Build the rule table
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
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
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
