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
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute the edge usage for each demand in each direction
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
    
    # base_load: load if all go clockwise
    # diff_load: difference (ccw - cw) per demand
    base_load = np.sum(load_cw, axis=0)
    diff_load = load_ccw - load_cw
    
    # Vectorized exhaustive search
    # Generate all 2^D bitmasks as a (2^D, D) binary matrix
    # Using bit shifting trick
    masks = np.arange(1 << D, dtype=np.uint32).reshape(-1, 1)
    bits = (masks >> np.arange(D, dtype=np.uint32)) & 1
    
    # Compute total loads for all masks
    # bits: (2^D, D), diff_load.T: (D, n)
    # Result: (2^D, n)
    all_loads = bits @ diff_load.T + base_load
    
    # Find the mask with the minimum maximum load
    max_loads = np.max(all_loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = best_idx
    
    # Convert best_mask back to routing array
    best_routing = np.zeros(D)
    for d in range(D):
        if (best_mask >> d) & 1:
            best_routing[d] = 1.0
    
    return best_routing


# EVOLVE-BLOCK-END
