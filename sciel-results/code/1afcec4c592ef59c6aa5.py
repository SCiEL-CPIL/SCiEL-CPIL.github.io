"""Construct a 5-regular graph of girth at least 9 using the Lazebnik-Ustimenko-Woldar family."""

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i,j) pairs.
    """
    q = k  # For k=5, we use GF(5)
    
    # Choose d so that girth >= d + 5 (for odd d)
    # We need girth >= g, so d + 5 >= g => d >= g - 5
    # For g=9, d >= 4. Use d=5 (odd) for girth >= 10
    d = g - 5
    if d < 3:
        d = 3
    # Ensure d is odd for the standard LUW construction
    if d % 2 == 0:
        d += 1
    
    # For q=5 (prime), field operations are simple modular arithmetic
    def f_add(a, b):
        return (a + b) % q
    
    def f_sub(a, b):
        return (a - b) % q
    
    def f_mul(a, b):
        return (a * b) % q
    
    # Recurrence rules for LUW construction
    # rules[c] = (a, b) means ncoords[c] = coords[c] +/- f_mul(ncoords[a], coords[b])
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates[r]
    
    # BFS over the incidence graph
    # Vertex label: (side, coords) where side in {0,1}, coords is tuple of length (d+1)
    # coords[0] is a sentinel (0), coords[1..d] are in GF(q)
    
    start_key = (0, (0,) * (d + 1))
    label = {start_key: 0}
    queue = [start_key]
    edge_list = []
    
    # Precompute field multiplication table for speed
    mul_table = [[(a * b) % q for b in range(q)] for a in range(q)]
    
    for vkey in queue:
        side, coords = vkey
        vidx = label[vkey]
        
        for first_val in range(q):
            # Build neighbor coordinates
            ncoords = [0] * (d + 1)
            ncoords[1] = first_val
            
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    # ncoords[c] = coords[c] + ncoords[a] * coords[b]
                    ncoords[c] = (coords[c] + mul_table[ncoords[a]][coords[b]]) % q
                else:
                    # ncoords[c] = coords[c] - coords[a] * ncoords[b]
                    ncoords[c] = (coords[c] - mul_table[coords[a]][ncoords[b]]) % q
            
            nkey = (1 - side, tuple(ncoords))
            
            if nkey not in label:
                label[nkey] = len(queue)
                queue.append(nkey)
            
            # Add edge (only from side 0 to avoid duplicates)
            if side == 0:
                edge_list.append((vidx, label[nkey]))
    
    return len(queue), edge_list
