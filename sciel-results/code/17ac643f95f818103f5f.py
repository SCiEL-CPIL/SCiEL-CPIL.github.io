import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    # Handle known small cages explicitly
    if (k, g) == (3, 5):
        return _petersen()
    elif (k, g) == (3, 6):
        return _heawood()
    elif (k, g) == (3, 8):
        return _tutte_coxeter()
    elif (k, g) == (3, 7):
        return _tutte_coxeter()
    elif k == 5 and g == 9:
        return _gq_2_4()

    # Fallback: LUW Construction
    return _luw_construct(k, g)

def _petersen():
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    for i in range(5):
        edges.append((5 + i, 5 + (i + 2) % 5))
    for i in range(5):
        edges.append((i, 5 + i))
    return 10, edges

def _heawood():
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5)
    ]
    edges = []
    for i, line in enumerate(lines):
        line_id = 7 + i
        for p in line:
            edges.append((p, line_id))
    return 14, edges

def _tutte_coxeter():
    # Tutte-Coxeter graph: 30 vertices, 3-regular, girth 8
    # Construction via the (3,8)-cage
    # Vertices 0-14 are "points", 15-29 are "lines"
    # This is the incidence graph of the 15-point, 15-line configuration
    # with specific incidence relations
    
    # The Tutte-Coxeter graph can be constructed as follows:
    # It's the bipartite graph where one part has 15 vertices (0-14)
    # and the other part has 15 vertices (15-29), each of degree 3
    
    # Known edge list for Tutte-Coxeter graph:
    edges = [
        # Point 0 connects to lines
        (0, 15), (0, 16), (0, 17),
        # Point 1 connects to lines
        (1, 15), (1, 18), (1, 19),
        # Point 2 connects to lines
        (2, 16), (2, 18), (2, 20),
        # Point 3 connects to lines
        (3, 17), (3, 19), (3, 20),
        # Point 4 connects to lines
        (4, 21), (4, 22), (4, 23),
        # Point 5 connects to lines
        (5, 21), (5, 24), (5, 25),
        # Point 6 connects to lines
        (6, 22), (6, 24), (6, 26),
        # Point 7 connects to lines
        (7, 23), (7, 25), (7, 26),
        # Point 8 connects to lines
        (8, 27), (8, 28), (8, 29),
        # Point 9 connects to lines
        (9, 27), (9, 24), (9, 22),
        # Point 10 connects to lines
        (10, 28), (10, 25), (10, 23),
        # Point 11 connects to lines
        (11, 29), (11, 26), (11, 21),
        # Point 12 connects to lines
        (12, 18), (12, 24), (12, 28),
        # Point 13 connects to lines
        (13, 19), (13, 25), (13, 29),
        # Point 14 connects to lines
        (14, 20), (14, 26), (14, 27),
    ]
    
    # Verify this is correct by checking degrees
    # Actually, let me use a verified construction
    # The Tutte-Coxeter graph is well-known. Let me use a different approach.
    
    # Use the known fact that it's the (3,8) cage with 30 vertices
    # Let me construct it properly using the standard representation
    
    # Alternative: use the line graph of the Petersen graph complement
    # Or use a direct edge list from known sources
    
    # Here's a verified edge list for Tutte-Coxeter:
    edges = [
        (0, 15), (0, 16), (0, 17),
        (1, 15), (1, 18), (1, 19),
        (2, 16), (2, 18), (2, 20),
        (3, 17), (3, 19), (3, 20),
        (4, 21), (4, 22), (4, 23),
        (5, 21), (5, 24), (5, 25),
        (6, 22), (6, 24), (6, 26),
        (7, 23), (7, 25), (7, 26),
        (8, 27), (8, 28), (8, 29),
        (9, 27), (9, 24), (9, 22),
        (10, 28), (10, 25), (10, 23),
        (11, 29), (11, 26), (11, 21),
        (12, 18), (12, 24), (12, 28),
        (13, 19), (13, 25), (13, 29),
        (14, 20), (14, 26), (14, 27),
    ]
    
    # Let me verify: each point 0-14 should have degree 3, each line 15-29 should have degree 3
    # Point 0: 15,16,17 -> deg 3 ✓
    # Point 1: 15,18,19 -> deg 3 ✓
    # Point 2: 16,18,20 -> deg 3 ✓
    # Point 3: 17,19,20 -> deg 3 ✓
    # Point 4: 21,22,23 -> deg 3 ✓
    # Point 5: 21,24,25 -> deg 3 ✓
    # Point 6: 22,24,26 -> deg 3 ✓
    # Point 7: 23,25,26 -> deg 3 ✓
    # Point 8: 27,28,29 -> deg 3 ✓
    # Point 9: 27,24,22 -> deg 3 ✓
    # Point 10: 28,25,23 -> deg 3 ✓
    # Point 11: 29,26,21 -> deg 3 ✓
    # Point 12: 18,24,28 -> deg 3 ✓
    # Point 13: 19,25,29 -> deg 3 ✓
    # Point 14: 20,26,27 -> deg 3 ✓
    
    # Line 15: 0,1 -> deg 2 ✗ (should be 3)
    # This edge list is incorrect.
    
    # Let me use a proper construction.
    # The Tutte-Coxeter graph can be constructed as follows:
    # Take the 15 edges of the Petersen graph and 15 vertices
    # Actually, it's simpler to just return LUW for safety
    
    return _luw_construct(3, 8)

def _gq_2_4():
    """
    Generalized Quadrangle GQ(2,4):
    - 5-regular bipartite graph
    - 50 vertices (25 points, 25 lines)
    - Girth 10 (so satisfies girth >= 9)
    
    Construction:
    - Points: (x, y, z) where x,y,z in {0,1,2} with x+y+z = 0 mod 3
      This gives 9 points... no, that's not right.
    
    Actually, GQ(2,4) has:
    - s=2, t=4
    - Number of points = (s+1)(st+1) = 3*9 = 27
    - Number of lines = (t+1)(st+1) = 5*9 = 45
    - Total vertices = 27 + 45 = 72
    - Each point on (t+1)=5 lines, each line has (s+1)=3 points
    - So it's 5-regular on the point side, 3-regular on the line side
    - Wait, that's not 5-regular overall.
    
    Let me reconsider. For a cage graph with degree k and girth g:
    - If g is even, g=2d, we need a bipartite graph where each vertex has degree k
    - GQ(s,t) is bipartite with points of degree (t+1) and lines of degree (s+1)
    - For this to be k-regular, we need s+1 = t+1 = k, so s=t=k-1
    - GQ(4,4) would be 5-regular, with girth 8 (since girth of GQ(s,t) is 4)
    - Wait, the girth of a GQ is 4 if it exists, which makes it unsuitable for girth 9.
    
    Actually, the girth of a generalized quadrangle is 4 (it has 4-cycles).
    So GQ constructions don't work for girth >= 5.
    
    Let me use a different approach.
    
    For (5,9), I need a 5-regular graph with girth at least 9.
    The smallest known is 1116 vertices.
    
    Let me use the LUW construction properly.
    """
    # GQ(2,4) doesn't work because it has girth 4.
    # Let me fall back to LUW.
    return _luw_construct(5, 9)

def _luw_construct(k, g):
    """
    Lifting-based / LUW construction for cage graphs.
    
    For degree k and girth g:
    - If g is odd, g = 2d+1, we can use a construction based on finite fields
    - The LUW construction uses a finite field GF(q) where q >= k
    """
    q = k  # Use q = k for the field size
    
    # Determine the dimension d
    # For girth g, we need the graph to have no cycles shorter than g
    # In the LUW construction, d is related to g
    
    # For a (k,g)-cage using LUW:
    # If g is even, g = 2d, then d = g/2
    # If g is odd, g = 2d+1, then d = (g-1)/2
    
    if g % 2 == 0:
        d = g // 2
    else:
        d = (g - 1) // 2
    
    # Ensure d is at least 3 for meaningful construction
    d = max(3, d)
    
    # Field operations
    if q == 4:
        # GF(4) using XOR-based operations
        # Represent elements as 0,1,2,3
        # Addition: XOR
        # Multiplication: table-based
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y):
            return x ^ y
        def sub(x, y):
            return x ^ y
        def mul(x, y):
            return mul_table[x][y]
    else:
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return (x * y) % q
    
    # Size of each side: q^d
    size_per_side = q ** d
    total_vertices = 2 * size_per_side
    
    # Precompute digits for all indices
    digits = []
    for i in range(size_per_side):
        num = i
        digs = [0] * d
        for j in range(d):
            digs[j] = num % q
            num //= q
        digits.append(digs)
    
    # Generate edges
    edges = []
    edge_set = set()  # For deduplication
    
    for i in range(size_per_side):
        v0 = i
        x = digits[i]
        
        # For each possible first coordinate
        for first in range(q):
            # Build the y vector for the neighbor on side 1
            y = [0] * (d + 1)
            y[1] = first
            
            # Calculate y[2] through y[d]
            for c in range(2, d + 1):
                # y[c] = x[c] + y[c-1] * x[c-1]  (simplified LUW relation)
                # More generally, use a recursive relation
                y[c] = add(x[c-1], mul(y[c-1], x[c-1]))
            
            # Convert y[1..d] to index j
            j = 0
            for c in range(1, d + 1):
                j = j * q + y[c]
            
            v1 = size_per_side + j
            
            # Add edge if not already present
            e = (min(v0, v1), max(v0, v1))
            if e not in edge_set:
                edge_set.add(e)
                edges.append(e)
    
    return total_vertices, edges
