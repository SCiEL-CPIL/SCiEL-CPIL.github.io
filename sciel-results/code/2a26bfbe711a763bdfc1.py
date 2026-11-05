"""Construct a 4-regular graph of girth 10 via the Lazebnik-Ustimenko-Woldar family."""
import time
import itertools

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses the LUG (Lazebnik-Ustimenko-Woldar) construction for prime q=k.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    q = k
    # For LUG(d, q), the girth is at least d + 5 if d is odd, d + 4 if d is even.
    # We want girth >= g.
    # If g is even (e.g., 10), we can pick d such that d+5 >= g (if d odd) or d+4 >= g (if d even).
    # Let's try to find the smallest d that works.
    # For g=10:
    #   d=5 (odd): girth >= 10. (5+5=10).
    #   d=6 (even): girth >= 10. (6+4=10).
    # d=5 gives a smaller graph typically? Let's check sizes.
    # Size of LUG(d, q) is roughly 2 * q^d.
    # d=5 -> 2*q^5. For q=4, 2*1024=2048.
    # d=4 (even): girth >= 8. Not enough for g=10.
    # So d=5 is the smallest for g=10.

    # Determine d
    d = None
    for candidate_d in range(3, 20):
        min_girth = candidate_d + 5 if candidate_d % 2 == 1 else candidate_d + 4
        if min_girth >= g:
            d = candidate_d
            break
    if d is None:
        d = 5 # fallback

    # Define field operations for GF(q)
    if q == 4:
        # GF(4) = {0, 1, 2, 3}
        # Addition is XOR
        # Multiplication table:
        #   0*0=0, 0*1=0, 0*2=0, 0*3=0
        #   1*0=0, 1*1=1, 1*2=2, 1*3=3
        #   2*0=0, 2*1=2, 2*2=3, 2*3=1
        #   3*0=0, 3*1=3, 3*2=1, 3*3=2
        add_table = [[i ^ j for j in range(4)] for i in range(4)]
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(a, b): return add_table[a][b]
        def sub(a, b): return add_table[a][b] # In char 2, sub == add
        def mul(a, b): return mul_table[a][b]
    else:
        def add(a, b): return (a + b) % q
        def sub(a, b): return (a - b) % q
        def mul(a, b): return (a * b) % q

    # The LUG construction rule depends on c (from 2 to d)
    # We need a rule[c] = (a, b) such that the graph is d-regular and has the required girth.
    # From Lazebnik et al., the rule is:
    # For c=2: (1, 1)
    # For c=3: (2, 1)
    # For c=4: (1, 2)
    # For c >= 5:
    #   Let r = (c-5) % 4
    #   Let c0 = c - r
    #   If r=0: (1, c0-2)
    #   If r=1: (c0-1, 1)
    #   If r=2: (c0, 1)
    #   If r=3: (1, c0+1)
    
    rule = {}
    rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        elif r == 3:
            rule[c] = (1, c0 + 1)

    # Vertices are pairs (side, tuple)
    # side 0: Points. tuple is (None, x2, x3, ..., xd) where xi in GF(q)
    # side 1: Lines.  tuple is (None, y2, y3, ..., yd) where yi in GF(q)
    # Actually, the standard representation uses vectors of length d+1 (indices 0..d).
    # Index 0 is the side marker or dummy. Indices 1..d are the coordinates.
    # Let's use indices 0..d for coordinates, but keep the side separate to avoid confusion.
    # Coordinate indices: 1, 2, ..., d.
    
    def neighbours(vertex):
        side, coords = vertex
        # coords is a tuple of length d+1 (index 0 unused or dummy, 1..d used)
        # We iterate over the free variable.
        # For a point (side 0), the free variable is typically y1 (the first coordinate of the line).
        # For a line (side 1), the free variable is typically x1 (the first coordinate of the point).
        
        if side == 0:
            # Point p. We want to find lines l such that p is incident to l.
            # The incidence relation: l[c] - p[c] = l[a] * p[b] for c=2..d.
            # Here l[1] is the free variable (let's call it t).
            # We can compute l[2..d] from p and t.
            for t in range(q):
                l_coords = [0] * (d + 1)
                l_coords[1] = t
                for c in range(2, d + 1):
                    a, b = rule[c]
                    # l[c] = p[c] + l[a] * p[b]
                    # Note: l[a] might be l[1] (t) or l[c'] for c' < c.
                    # Since a < c always in the rule above?
                    # c=2: (1,1) -> a=1. OK.
                    # c=3: (2,1) -> a=2. OK.
                    # c=4: (1,2) -> a=1. OK.
                    # c=5: (1, 3) -> a=1. OK.
                    # c=6: (5, 1) -> a=5. OK.
                    # c=7: (6, 1) -> a=6. OK.
                    # c=8: (1, 7) -> a=1. OK.
                    # c=9: (8, 1) -> a=8. OK.
                    # c=10: (9, 1) -> a=9. OK.
                    # c=11: (1, 10) -> a=1. OK.
                    # Yes, a is always < c. So l[a] is already computed.
                    l_coords[c] = add(coords[c], mul(l_coords[a], coords[b]))
                yield (1, tuple(l_coords))
        else:
            # Line l. We want to find points p such that l is incident to p.
            # p[c] = l[c] - l[a] * p[b] ? No, the relation is symmetric in a sense but let's derive.
            # l[c] - p[c] = l[a] * p[b]  =>  p[c] = l[c] - l[a] * p[b]
            # Free variable is p[1] (let's call it t).
            for t in range(q):
                p_coords = [0] * (d + 1)
                p_coords[1] = t
                for c in range(2, d + 1):
                    a, b = rule[c]
                    # p[c] = l[c] - l[a] * p[b]
                    p_coords[c] = sub(coords[c], mul(coords[a], p_coords[b]))
                yield (0, tuple(p_coords))

    # Build the graph via BFS/DFS from the zero point
    start_vertex = (0, tuple([0] * (d + 1)))
    index_map = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    
    # We need to ensure we capture all edges. Since it's bipartite, we can just add edges when visiting from side 0.
    # Or add all edges and then deduplicate.
    # Let's add edges as we go.
    
    for vertex in order:
        # Check time
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index_map:
                index_map[other] = len(order)
                order.append(other)
            # Add edge if not already added.
            # To avoid duplicates, only add if vertex is on side 0 (points).
            if vertex[0] == 0:
                edges.append((index_map[vertex], index_map[other]))

    # The graph might be disconnected if q is not prime or if the construction fails to connect.
    # For prime q, LUG(d,q) is connected.
    # For q=4, it should also be connected for d>=3?
    # Let's assume it's connected. If not, we return the largest component?
    # The task says "The connected component of the zero point is returned" in the seed.
    # So we just return what we found.

    n = len(order)
    return n, edges

# EVOLVE-BLOCK-END
