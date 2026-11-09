import time
import numpy as np

def _luw_construct_numpy(k, g):
    """Optimized LUW construction using NumPy."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: np.vectorize(lambda a, b: table[a][b], otypes=[int])(x, y)
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    qd = q ** d
    total = 2 * qd
    
    # Generate all vertices
    # Vertex v: side = v // qd, rem = v % qd
    # Decode rem into digits y[1]...y[d]
    
    sides = np.arange(total) // qd
    rems = np.arange(total) % qd
    
    # Decode rems into digit arrays
    # y[c] = (rem // q^(d-c)) % q
    y = np.zeros((total, d), dtype=np.int32)
    for c in range(d):
        y[:, c] = (rems // (q ** (d - 1 - c))) % q
    
    # For each vertex and each 'first' value, compute neighbor
    all_neighbors = []
    for first in range(q):
        y_new = np.zeros((total, d), dtype=np.int32)
        y_new[:, 0] = first
        
        for c in range(1, d):
            a, b = rule[c + 1]
            a_idx = a - 1
            b_idx = b - 1
            if c + 1 <= d:
                if sides[0] == 0:  # This is wrong - sides varies per vertex
                    pass
        
        # Need to handle side-dependent computation
        # For side=0: y_new[c] = y[c] + y_new[a] * y[b]
        # For side=1: y_new[c] = y[c] - y[a] * y_new[b]
        
        # Process in two passes: side=0 and side=1
        mask0 = (sides == 0)
        mask1 = (sides == 1)
        
        y_new0 = np.zeros((np.sum(mask0), d), dtype=np.int32)
        y_new0[:, 0] = first
        y0 = y[mask0]
        for c in range(1, d):
            a, b = rule[c + 1]
            y_new0[:, c] = (y0[:, c] + y_new0[:, a-1] * y0[:, b-1]) % q
        
        y_new1 = np.zeros((np.sum(mask1), d), dtype=np.int32)
        y_new1[:, 0] = first
        y1 = y[mask1]
        for c in range(1, d):
            a, b = rule[c + 1]
            y_new1[:, c] = (y1[:, c] - y1[:, a-1] * y_new1[:, b-1]) % q
        
        # Encode neighbors
        # nb_rem = sum(y_new[c] * q^(d-1-c))
        powers = (q ** np.arange(d - 1, -1, -1)).astype(np.int64)
        
        nb_rems0 = (y_new0 * powers).sum(axis=1)
        nb_rems1 = (y_new1 * powers).sum(axis=1)
        
        nb0 = 1 * qd + nb_rems0  # side flips from 0 to 1
        nb1 = 0 * qd + nb_rems1  # side flips from 1 to 0
        
        # Map back to original indices
        idx0 = np.where(mask0)[0]
        idx1 = np.where(mask1)[0]
        
        nbs = np.zeros(total, dtype=np.int64)
        nbs[idx0] = nb0
        nbs[idx1] = nb1
        
        all_neighbors.append(nbs)
    
    # Build edge list
    edges = []
    seen = set()
    for first in range(q):
        nbs = all_neighbors[first]
        for v in range(total):
            nb = int(nbs[v])
            if v < nb:
                edge = (v, nb)
                if edge not in seen:
                    seen.add(edge)
                    edges.append(edge)
    
    return total, edges
