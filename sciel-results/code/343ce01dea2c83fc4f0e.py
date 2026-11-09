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
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute edge masks for each demand in both directions
    # clockwise_mask[d] : bitmask of edges used if demand d goes clockwise
    # ccw_mask[d]       : bitmask of edges used if demand d goes counter-clockwise
    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        cw_edges = set()
        e = s
        while e != t:
            cw_edges.add(e)
            e = (e + 1) % n
        ccw_edges = set(range(n)) - cw_edges
        cw_mask = 0
        for e in cw_edges:
            cw_mask |= (1 << e)
        ccw_mask = 0
        for e in ccw_edges:
            ccw_mask |= (1 << e)
        cw_masks.append(cw_mask)
        ccw_masks.append(ccw_mask)

    weights = np.array([w for s, t, w in demands], dtype=np.float64)

    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=np.float64)

    # Exhaustive search over all 2^15 combinations
    for mask in range(1 << D):
        loads = np.zeros(n, dtype=np.float64)
        for d in range(D):
            if (mask >> d) & 1:
                # counter-clockwise
                m = ccw_masks[d]
            else:
                # clockwise
                m = cw_masks[d]
            # Add weight to each edge in the mask
            for e in range(n):
                if (m >> e) & 1:
                    loads[e] += weights[d]
        
        ml = np.max(loads)
        if ml < best_max_load:
            best_max_load = ml
            best_routing = np.array([(mask >> d) & 1 for d in range(D)], dtype=np.float64)

    return best_routing


# EVOLVE-BLOCK-END
