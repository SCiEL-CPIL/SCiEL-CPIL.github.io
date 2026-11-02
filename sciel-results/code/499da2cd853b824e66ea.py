<<<<<<< SEARCH
def _luw_construct(k, d):
    """
    Lazebnik-Ustimenko-Woldar construction CD(d, k) with q = k.
    Returns a k-regular bipartite graph on 2*k^d vertices.
    """
    q = k
    
    # Generate the rule for c = 2..d
    rule = {}
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Precompute multiplication table
    mul_table = [[(i * j) % q for j in range(q)] for i in range(q)]
    
    # Precompute all vertices as flat indices
    # Side 0: x[1..d] each in 0..q-1 -> q^d vertices
    # Side 1: x[1..d] each in 0..q-1 -> q^d vertices
    # Total: 2*q^d vertices
    
    # Map from (side, coord_tuple) to flat index
    # For side 0: index = coord[0]*q^(d-1) + coord[1]*q^(d-2) + ... + coord[d-1]
    # For side 1: index = q^d + coord[0]*q^(d-1) + ...
    
    q_powers = [1] * (d + 1)
    for i in range(1, d + 1):
        q_powers[i] = q_powers[i-1] * q
    
    def coord_to_index(side, coords):
        """coords is a tuple of d elements, each 0..q-1"""
        idx = 0
        for i in range(d):
            idx += coords[i] * q_powers[d - 1 - i]
        if side == 1:
            idx += q_powers[d]
        return idx
    
    # Precompute neighbors for all vertices
    # For a vertex on side 0 with coordinates x[1..d]:
    #   For each first in 0..q-1:
    #     y[1] = first
    #     For c in 2..d: y[c] = x[c] + y[a] * x[b] mod q
    #     Neighbor is on side 1 with coordinates y[1..d]
    #
    # For a vertex on side 1 with coordinates x[1..d]:
    #   For each first in 0..q-1:
    #     y[1] = first
    #     For c in 2..d: y[c] = x[c] - x[a] * y[b] mod q
    #     Neighbor is on side 0 with coordinates y[1..d]
    
    n = 2 * q_powers[d]
    adj = [[] for _ in range(n)]
    
    # Precompute for side 0 vertices
    for x0 in range(q):
        for x1 in range(q):
            # ... iterate over all q^d combinations
            pass
    
    # More efficient: iterate over all vertices
    for idx in range(n):
        if idx < q_powers[d]:
            side = 0
            flat = idx
        else:
            side = 1
            flat = idx - q_powers[d]
        
        # Decode coordinates
        coords = [0] * d
        for i in range(d - 1, -1, -1):
            coords[i] = flat % q
            flat //= q
        
        # Compute neighbors
        for first in range(q):
            y = [0] * (d + 1)  # 1-indexed: y[1..d]
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = (coords[c-1] + mul_table[y[a]][coords[b-1]]) % q
                else:
                    y[c] = (coords[c-1] - mul_table[coords[a-1]][y[b]]) % q
            other_side = 1 - side
            other_flat = 0
            for i in range(d):
                other_flat += y[i+1] * q_powers[d - 1 - i]
            other_idx = coord_to_index(other_side, tuple(y[1:d+1]))
            adj[idx].append(other_idx)
    
    # Build edge list (only one direction to avoid duplicates)
    edges = []
    for u in range(q_powers[d]):  # Only side 0 vertices
        for v in adj[u]:
            edges.append((u, v))
    
    return n, edges
=======
def _luw_construct_fast(k, d):
    """
    Fast Lazebnik-Ustimenko-Woldar construction CD(d, k) with q = k.
    Returns a k-regular bipartite graph on 2*k^d vertices.
    Uses precomputed neighbor tables for speed.
    """
    q = k
    
    # Generate the rule for c = 2..d
    rule = {}
    if d >= 2:
        rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    # Precompute multiplication table
    mul_table = [[(i * j) % q for j in range(q)] for i in range(q)]
    
    q_powers = [1] * (d + 1)
    for i in range(1, d + 1):
        q_powers[i] = q_powers[i-1] * q
    
    qd = q_powers[d]
    n = 2 * qd
    
    # Precompute all coordinate tuples for side 0
    # For each flat index, store the coordinate tuple
    all_coords = []
    for flat in range(qd):
        coords = [0] * d
        tmp = flat
        for i in range(d - 1, -1, -1):
            coords[i] = tmp % q
            tmp //= q
        all_coords.append(tuple(coords))
    
    # Precompute neighbors for all side-0 vertices
    # adj_side0[u] = list of side-1 neighbor indices (offset by qd)
    adj_side0 = [[] for _ in range(qd)]
    
    for u in range(qd):
        coords = all_coords[u]
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = (coords[c-1] + mul_table[y[a]][coords[b-1]]) % q
            # Convert y[1..d] to flat index on side 1
            other_flat = 0
            for i in range(d):
                other_flat += y[i+1] * q_powers[d - 1 - i]
            adj_side0[u].append(qd + other_flat)
    
    # Build edge list from side 0 only
    edges = []
    for u in range(qd):
        for v in adj_side0[u]:
            edges.append((u, v))
    
    return n, edges
>>>>>>> REPLACE

