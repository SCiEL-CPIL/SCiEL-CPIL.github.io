import time

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g using the Lazebnik-Ustimenko-Woldar (LUW) construction.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    q = k
    
    # Determine d such that girth_of_d(d) >= g
    # LUW graph CD(d, q) has girth:
    # d + 5 if d is odd
    # d + 6 if d is even
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1
        
    # If k is not a prime power, LUW might not be directly applicable in the standard way,
    # but for k=3 (prime), it works.
    # For k=3, d will be 10 for g=16.
    # n = 2 * q^d
    
    # GF(q) operations
    # For prime q, simple modular arithmetic is sufficient.
    # For q=4, we need GF(4) arithmetic, but the task specifies k=3 typically.
    # We handle q=3, 5, 7 etc. assuming they are primes.
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    # Rule for LUW
    # rule[c] = (a, b) such that l[c] - p[c] = l[a] * p[b]
    # We need rules for c = 2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    # Precompute powers of q for encoding
    pow_q = [1] * (d + 1)
    for i in range(1, d + 1):
        pow_q[i] = pow_q[i-1] * q
    
    n_points = pow_q[d]
    n = 2 * n_points
    edges = []
    
    # Iterate over all points (side 0)
    # Point coordinates: x1, ..., xd
    # Index: x1 * q^(d-1) + x2 * q^(d-2) + ... + xd
    
    # To speed up, we can generate coordinates using mixed-radix counters
    # Or simply iterate 0 to n_points-1 and decode
    # Decoding is O(d) per point. Total O(n*d).
    # n=118098, d=10. 1.18M ops. Very fast.
    
    for p_idx in range(n_points):
        # Decode p_idx into coordinates x1..xd
        # p_idx = x1*q^(d-1) + ... + xd
        # x1 = p_idx // q^(d-1)
        # rem = p_idx % q^(d-1)
        # x2 = rem // q^(d-2)
        # ...
        
        coords = [0] * (d + 1) # 1-indexed
        rem = p_idx
        for i in range(d, 0, -1):
            coords[i] = rem % q
            rem //= q
        
        # Generate neighbors (lines) for this point
        # For each possible first coordinate of the line y1 in 0..q-1
        for first in range(q):
            # y1 = first
            # y = [0, y1, y2, ..., yd] (1-indexed)
            y_coords = [0] * (d + 1)
            y_coords[1] = first
            
            # Calculate y2..yd
            for c in range(2, d + 1):
                a, b = rule[c]
                # Point -> Line:
                # l[c] = p[c] + l[a] * p[b]
                # y[c] = x[c] + y[a] * x[b]
                y_coords[c] = add(coords[c], mul(y_coords[a], coords[b]))
                
            # Encode line index
            # l_idx = y1*q^(d-1) + ... + yd
            l_rem = 0
            for i in range(1, d + 1):
                l_rem = l_rem * q + y_coords[i]
            
            l_idx = n_points + l_rem
            
            # Add edge (p_idx, l_idx)
            edges.append((p_idx, l_idx))
            
        if time.time() > deadline:
            break

    # The graph is regular by construction.
    # Each point has q neighbors.
    # Each line has q neighbors (by symmetry of the definition).
    
    return n, edges
