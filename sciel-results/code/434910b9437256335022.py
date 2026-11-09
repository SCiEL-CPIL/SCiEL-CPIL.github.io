# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges.
    Uses an exhaustive search over all 2^15 possible routing combinations to find
    the one that minimizes the maximum edge load.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin for 600s limit

    n_nodes = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]
    D = len(demands)

    # Precompute the edge sets for clockwise and counter-clockwise routes
    # cw_matrix[i][j] = 1 if demand i uses edge j when routed clockwise
    # ccw_matrix[i][j] = 1 if demand i uses edge j when routed counter-clockwise
    cw_matrix = np.zeros((D, n_nodes), dtype=np.float64)
    ccw_matrix = np.zeros((D, n_nodes), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise: edges s, s+1, ..., t-1 (mod n)
        e = s
        while e != t:
            cw_matrix[i, e] = 1.0
            e = (e + 1) % n_nodes
        
        # Counter-clockwise: all other edges
        # Since the ring is complete, ccw edges are the complement of cw edges
        # We can compute ccw_matrix as 1 - cw_matrix because every edge is either in CW or CCW path
        # Wait, is that true?
        # Clockwise path: s -> s+1 -> ... -> t
        # Counter-clockwise path: s -> s-1 -> ... -> t
        # The union of these two paths is the entire ring (all 10 edges).
        # They are disjoint (except for nodes, but we are dealing with edges).
        # Edge e is in CW path if traversing from s to t clockwise crosses e.
        # Edge e is in CCW path if traversing from s to t counter-clockwise crosses e.
        # Yes, for a ring, the two paths partition the set of edges.
        # So ccw_matrix = 1 - cw_matrix.
        
    ccw_matrix = 1.0 - cw_matrix
    
    weights = np.array([w for _, _, w in demands], dtype=np.float64)

    # Base loads if all demands are routed clockwise
    # loads_cw[j] = sum_i weights[i] * cw_matrix[i, j]
    base_loads = weights @ cw_matrix
    
    # Difference in loads when switching demand i from CW to CCW
    # delta_loads[i, j] = weights[i] * (ccw_matrix[i, j] - cw_matrix[i, j])
    # Note: ccw_matrix[i, j] - cw_matrix[i, j] is 1 if edge j is in CCW path, -1 if in CW path, 0 if neither (impossible here)
    diff_matrix = weights[:, np.newaxis] * (ccw_matrix - cw_matrix)
    
    # We want to find routing vector r (0 or 1) that minimizes max(base_loads + diff_matrix.T @ r)
    # Let M = diff_matrix.T (shape n_nodes x D)
    # loads = base_loads + M @ r
    
    M = diff_matrix.T
    
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=np.float64)
    
    # Exhaustive search over all 2^15 combinations
    # 2^15 = 32768 iterations
    for mask in range(1 << D):
        # Check deadline periodically
        if (mask & 0xFF) == 0 and time.time() > deadline:
            break
            
        # Convert mask to routing vector r
        # r[i] = 1 if bit i is set, else 0
        # We can use bit operations or numpy
        # For speed, we can precompute routing vectors or use a trick
        # But 32768 is small, so simple conversion is fine.
        # r = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.float64)
        
        # Faster: use np.unpackbits or similar? No, D=15 is small.
        # Let's just do the list comprehension, it's fast enough for 32k iterations.
        r = np.zeros(D, dtype=np.float64)
        m = mask
        for i in range(D):
            if m & 1:
                r[i] = 1.0
            m >>= 1
            
        # Compute loads
        loads = base_loads + M @ r
        max_load = np.max(loads)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = r.copy()
            
    return best_routing

# EVOLVE-BLOCK-END
