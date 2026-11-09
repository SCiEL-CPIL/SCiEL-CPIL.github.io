# EVOLVE-BLOCK-START
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
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
        (5, 2, 2)
    ]
    
    D = len(demands)
    
    # Precompute CW and CCW weight vectors for each demand
    cw_vectors = np.zeros((D, n))
    ccw_vectors = np.zeros((D, n))
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise path edges
        if s <= t:
            cw_vectors[i, s:t] = w
        else:
            cw_vectors[i, s:] = w
            cw_vectors[i, :t] = w
        
        # Counter-clockwise path edges are the complement
        ccw_vectors[i] = np.where(cw_vectors[i] == 0, w, 0)
        
    # Vectorized Exhaustive Search
    # Base load if all demands go CW
    base_load = np.sum(cw_vectors, axis=0)
    
    # Difference in load if demand i goes CCW instead of CW
    # Load_i = CW_i if M_i=0, CCW_i if M_i=1
    # Load_i = CW_i + M_i * (CCW_i - CW_i)
    diff = ccw_vectors - cw_vectors
    
    # Generate all 2^D masks as a matrix of bits
    # masks_int: shape (2^D,)
    masks_int = np.arange(1 << D, dtype=np.uint8)
    # bits: shape (2^D, D)
    # We want bit i of mask to be in column i
    # np.arange(D) is [0, 1, ..., D-1]
    bits = (masks_int[:, None] >> np.arange(D)[None, :]) & 1
    
    # Compute loads for all masks
    # Loads shape: (2^D, n)
    # Loads[k, e] = base_load[e] + sum_i bits[k, i] * diff[i, e]
    loads = base_load[None, :] + (bits @ diff)
    
    # Find the mask with the minimum maximum load
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    
    # Extract best mask
    best_mask = masks_int[best_idx]
    
    # Construct routing array
    routing = np.zeros(D)
    for i in range(D):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing

# EVOLVE-BLOCK-END
