# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        value < 0.5 -> route the demand clockwise
        value >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    start_time = time.time()
    deadline = 600.0  # Safety margin from 630s limit
    
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute the edge usage for each demand in each direction
    # clockwise_mask[d]: boolean array of length n, True if edge e is used when routing demand d clockwise
    # ccw_mask[d]: boolean array of length n, True if edge e is used when routing demand d counter-clockwise
    cw_masks = np.zeros((D, n), dtype=np.int8)
    ccw_masks = np.zeros((D, n), dtype=np.int8)
    
    for d, (s, t, w) in enumerate(demands):
        # Clockwise: edges s, s+1, ..., t-1 (mod n)
        if s <= t:
            for e in range(s, t):
                cw_masks[d, e] = 1
        else:
            for e in range(s, n):
                cw_masks[d, e] = 1
            for e in range(0, t):
                cw_masks[d, e] = 1
        
        # Counter-clockwise: all other edges
        ccw_masks[d] = 1 - cw_masks[d]
    
    # Weights
    weights = np.array([w for (_, _, w) in demands], dtype=np.float64)
    
    # Precompute the load vector contribution for each demand in each direction
    load_cw = weights[:, np.newaxis] * cw_masks
    load_ccw = weights[:, np.newaxis] * ccw_masks
    
    # Base load if all go clockwise
    base_load = np.sum(load_cw, axis=0)
    # Difference in load if we switch demand d to counter-clockwise
    diff_load = load_ccw - load_cw
    
    best_max_load = np.inf
    best_routing = np.zeros(D)
    
    # Exhaustive search over all 2^15 = 32768 combinations
    # This is fast enough (32768 * 10 operations is trivial)
    # We iterate through all bitmasks
    
    for mask in range(1 << D):
        if time.time() > deadline:
            break
            
        # Compute total load for this configuration
        # total_load = base_load + sum_{d: bit d is 1} diff_load[d]
        total_load = base_load.copy()
        m = mask
        for d in range(D):
            if m & 1:
                total_load += diff_load[d]
            m >>= 1
        
        max_load = np.max(total_load)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = np.zeros(D)
            m = mask
            for d in range(D):
                if m & 1:
                    best_routing[d] = 1.0
                m >>= 1

    return best_routing

# EVOLVE-BLOCK-END
