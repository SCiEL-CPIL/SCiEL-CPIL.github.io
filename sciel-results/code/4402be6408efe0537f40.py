"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _moore_bound(k, g):
    """Compute the Moore bound (lower bound) for a k-regular graph of girth g."""
    if k <= 2:
        return g if g % 2 == 0 else 2 * g - 1
    if g % 2 == 0:
        half = g // 2
        s = 0
        term = 1
        for i in range(half):
            s += term
            term *= (k - 1)
        return 2 * k * s
    else:
        half = (g - 1) // 2
        s = 0
        term = 1
        for i in range(half + 1):
            s += term
            term *= (k - 1)
        return 1 + k * s

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using the Lazebnik-Ustimenko-Woldar
    (LUW) family of graphs. For k=5, g=9, this produces a 5-regular bipartite graph
    with girth 10 (which satisfies girth >= 9).
    """
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill

    q = k  # For k=5, q=5 (prime)

    # Choose d so that the girth is at least g
    # For the LUW family, the girth is approximately d + 4 or d + 5
    # For odd d: girth >= d + 5
    # For even d: girth >= d + 4
    # We need girth >= g, so:
    #   If d is odd: d + 5 >= g  =>  d >= g - 5
    #   If d is even: d + 4 >= g  =>  d >= g - 4
    
    # Find the smallest d that satisfies the girth requirement
    d = None
    for candidate_d in range(3, 20):
        if candidate_d % 2 == 1:  # odd
            if candidate_d + 5 >= g:
                d = candidate_d
                break
        else:  # even
            if candidate_d + 4 >= g:
                d = candidate_d
                break
    
    if d is None:
        d = g - 4  # Fallback
    
    # Ensure d is at least 3
    if d < 3:
        d = 3

    # Finite-field helpers
    if q == 4:
        _add_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        _mul_t = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        def f_add(a, b): return a ^ b
        def f_sub(a, b): return a ^ b
        def f_mul(a, b): return _mul_t[a][b]
    else:
        def f_add(a, b): return (a + b) % q
        def f_sub(a, b): return (a - b) % q
        def f_mul(a, b): return (a * b) % q
    
    # Recurrence rule for the LUW construction
    # rules[c] = (a, b) means ncoords[c] = coords[c] + ncoords[a] * coords[b] (side 0)
    # or ncoords[c] = coords[c] - coords[a] * ncoords[b] (side 1)
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates[r]
    
    # BFS over the incidence graph; label vertices by their coordinate tuple.
    # A vertex is (side, coords) where side in {0,1} and coords is a length-(d+1)
    # tuple with coords[0] ignored (None sentinel) and coords[1..d] in GF(q).
    start_key = (0, (0,) * (d + 1))
    label = {start_key: 0}
    queue = [start_key]
    edge_list = []
    
    for vkey in queue:
        if time.time() > deadline:
            break
        side, coords = vkey
        vidx = label[vkey]
        for first_val in range(q):
            # Build the neighbour coordinate array
            ncoords = [0] * (d + 1)
            ncoords[1] = first_val
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                else:
                    ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
            nkey = (1 - side, tuple(ncoords))
            if nkey not in label:
                label[nkey] = len(queue)
                queue.append(nkey)
            if side == 0:
                edge_list.append((vidx, label[nkey]))
    
    n = len(queue)
    return n, edge_list
# EVOLVE-BLOCK-END
