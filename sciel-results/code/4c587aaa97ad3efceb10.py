# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Uses Gray-code incremental update for O(n) per step exhaustive search.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute for each demand:
    #   cw_edges[d]: set of edges used if routed clockwise
    #   ccw_edges[d]: set of edges used if routed counter-clockwise
    cw_mask = np.zeros((D, n), dtype=np.float64)
    weights = np.zeros(D)
    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        if s == t:
            continue
        e = s
        while e != t:
            cw_mask[d, e] = 1.0
            e = (e + 1) % n

    # Weighted CW contribution per demand per edge
    cw_w = cw_mask * weights[:, None]

    # base_load: load if all demands go CW
    base_load = cw_w.sum(axis=0)

    # diff[d, e] = ccw_w[d,e] - cw_w[d,e] = w*(1-cw_mask[d,e]) - w*cw_mask[d,e]
    #           = w*(1 - 2*cw_mask[d,e])
    diff = weights[:, None] * (1.0 - 2.0 * cw_mask)

    # Exhaustive search using Gray code for incremental updates
    # Gray code: consecutive states differ in exactly one bit
    total_states = 1 << D
    best_routing = np.zeros(D, dtype=np.int8)
    best_max_load = np.inf

    # Current load starts at base_load (all CW, r = 0)
    load = base_load.copy()
    current_bits = 0

    for i in range(total_states):
        # Gray code: g = i ^ (i >> 1)
        g = i ^ (i >> 1)
        
        if i == 0:
            # First state: all zeros, load = base_load
            max_load = load.max()
            if max_load < best_max_load:
                best_max_load = max_load
                best_routing = np.zeros(D, dtype=np.int8)
            current_bits = g
            continue
        
        # Find which bit changed between previous Gray code and current
        prev_g = (i - 1) ^ ((i - 1) >> 1)
        changed = g ^ prev_g
        d = changed.bit_length() - 1  # index of the changed bit
        
        # Determine if bit d went from 0 to 1 or 1 to 0
        if (g >> d) & 1:
            # Bit d turned ON (CW -> CCW): add diff[d]
            load += diff[d]
        else:
            # Bit d turned OFF (CCW -> CW): subtract diff[d]
            load -= diff[d]
        
        current_bits = g
        max_load = load.max()
        if max_load < best_max_load:
            best_max_load = max_load
            # Extract routing from Gray code bits
            best_routing = np.array([(g >> d2) & 1 for d2 in range(D)], dtype=np.int8)

    return best_routing.astype(float)


# EVOLVE-BLOCK-END
