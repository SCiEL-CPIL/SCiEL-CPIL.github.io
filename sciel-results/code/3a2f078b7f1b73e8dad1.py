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
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    # Deadline guard
    start_time = time.time()
    DEADLINE = start_time + 600.0

    n_nodes = 10
    n_edges = 10
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

    # Precompute clockwise edge masks for each demand
    cw_masks = []
    for s, t, w in demands:
        mask = np.zeros(n_edges, dtype=np.int8)
        if s <= t:
            for e in range(s, t):
                mask[e] = 1
        else:
            for e in range(s, n_nodes):
                mask[e] = 1
            for e in range(0, t):
                mask[e] = 1
        cw_masks.append(mask)
    
    cw_masks = np.array(cw_masks) # Shape (D, n_edges)
    ccw_masks = 1 - cw_masks
    weights = np.array([w for _, _, w in demands], dtype=np.int64)
    deltas = weights[:, None] * (ccw_masks - cw_masks)
    
    # We use an incremental update strategy with Gray code enumeration.
    # State: current routing vector x (0 or 1 for each demand)
    # Load: current load vector
    
    # Initialize with all clockwise (x = 0)
    x = np.zeros(D, dtype=np.int64)
    current_load = np.sum(weights[:, None] * cw_masks, axis=0).astype(np.int64)
    best_load = np.max(current_load)
    best_routing = x.copy()
    
    # Gray code sequence for 15 bits
    # We iterate through all 2^15 states.
    # Gray code ensures only one bit changes at a time.
    
    for i in range(1, 1 << D):
        if time.time() > DEADLINE:
            break
            
        # Find bit index to flip
        bit_idx = 0
        temp = i
        while (temp & 1) == 0:
            temp >>= 1
            bit_idx += 1
            
        # Update x
        if x[bit_idx] == 0:
            x[bit_idx] = 1
            current_load += deltas[bit_idx]
        else:
            x[bit_idx] = 0
            current_load -= deltas[bit_idx]
            
        max_load = np.max(current_load)
        if max_load < best_load:
            best_load = max_load
            best_routing = x.copy()
            
    return best_routing.astype(np.float64)

# EVOLVE-BLOCK-END
