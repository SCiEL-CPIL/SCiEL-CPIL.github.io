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
        value >= 0.5 -> route the demand counter-clockwise
        value < 0.5  -> route the demand clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute for each demand:
    #   cw_mask[d, e] = 1 if edge e is used when routing demand d clockwise
    #   ccw_mask[d, e] = 1 if edge e is used when routing demand d counter-clockwise
    cw_mask = np.zeros((D, n), dtype=np.int32)
    ccw_mask = np.zeros((D, n), dtype=np.int32)
    weights = np.zeros(D, dtype=np.float64)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        if s == t:
            continue
        e = s
        while e != t:
            cw_mask[d, e] = 1
            e = (e + 1) % n
        # Counter-clockwise uses all other edges
        for e in range(n):
            if cw_mask[d, e] == 0:
                ccw_mask[d, e] = 1

    # delta[d, e] = (ccw_mask[d, e] - cw_mask[d, e]) * weights[d]
    # If we flip demand d to CCW, the load on edge e changes by delta[d, e]
    delta = (ccw_mask - cw_mask) * weights[:, np.newaxis]

    # Base load when all demands are CW
    base_load = (weights[:, np.newaxis] * cw_mask).sum(axis=0)

    # Vectorized exact search over all 2^D combinations
    # D = 15, so 2^15 = 32768 masks
    # We can represent all masks as a binary matrix M of shape (2^D, D)
    # M[i, d] = 1 if the d-th bit of i is set (meaning demand d is routed CCW)
    
    # Efficiently generate binary matrix for all masks
    # Using np.arange and bit shifting
    masks = np.arange(1 << D, dtype=np.int32)
    
    # We need M where M[i, d] = (masks[i] >> d) & 1
    # This can be done with broadcasting
    # bits: shape (D, 1) containing 1 << d
    # masks: shape (1 << D, 1)
    # M = ((masks[:, None] & bits[None, :]) != 0).astype(np.float64)
    
    bits = (1 << np.arange(D, dtype=np.int32))
    M = ((masks[:, None] & bits[None, :]) != 0).astype(np.float64)
    
    # Compute loads for all masks at once
    # loads[i, e] = base_load[e] + sum_d M[i, d] * delta[d, e]
    # loads = base_load[None, :] + M @ delta
    
    loads = base_load[None, :] + M @ delta
    
    # Find the mask with the minimum maximum load
    max_loads = loads.max(axis=1)
    best_mask_idx = np.argmin(max_loads)
    
    # Extract the best routing
    # M[best_mask_idx] is a vector of 0s and 1s indicating which demands are CCW
    best_routing = M[best_mask_idx]
    
    # The return value should be an array where >= 0.5 means CCW.
    # Since best_routing is already 0.0 or 1.0, we can return it directly.
    # However, the problem statement says "one per demand in that order".
    # Our M matrix is constructed such that column d corresponds to demand d.
    # So M[best_mask_idx] is the correct vector.
    
    return best_routing

# EVOLVE-BLOCK-END
