# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Strategy:
      1. Compute the exact optimum via exhaustive enumeration of all 2^15 = 32768
         routing assignments using a precomputed incremental-load table.
      2. Return the routing corresponding to the assignment with minimum max-load.

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
        0.0 -> clockwise, 1.0 -> counter-clockwise
    """
    n = 10
    D = 15

    # Demands (source, target, weight) in order
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

    # Precompute edge indices for clockwise and counter-clockwise routes
    cw_edges = []
    ccw_edges = []

    for (s, t, w) in demands:
        cw = []
        if s <= t:
            cw = list(range(s, t))
        else:
            cw = list(range(s, n)) + list(range(0, t))
        cw_edges.append(cw)

        cw_set = set(cw)
        ccw = [e for e in range(n) if e not in cw_set]
        ccw_edges.append(ccw)

    weights = np.array([d[2] for d in demands], dtype=np.float64)

    # Contribution vectors
    cw_contrib = np.zeros((D, n), dtype=np.float64)
    ccw_contrib = np.zeros((D, n), dtype=np.float64)

    for d in range(D):
        w = weights[d]
        for e in cw_edges[d]:
            cw_contrib[d, e] = w
        for e in ccw_edges[d]:
            ccw_contrib[d, e] = w

    # Delta vector: change in edge loads when flipping demand d from CW to CCW
    delta = ccw_contrib - cw_contrib  # shape (D, n)

    # Base load: all demands routed clockwise
    base_load = np.sum(cw_contrib, axis=0)  # shape (n,)

    # Enumerate all 2^D masks using incremental DP
    num_masks = 1 << D
    load_table = np.empty((num_masks, n), dtype=np.float64)
    load_table[0] = base_load

    for mask in range(1, num_masks):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        prev = mask ^ lsb
        load_table[mask] = load_table[prev] + delta[i]

    # Vectorized: max load per mask
    max_loads = np.max(load_table, axis=1)

    # Optimal mask
    best_mask = int(np.argmin(max_loads))

    # Build routing array
    routing = np.zeros(D, dtype=np.float64)
    for d in range(D):
        if (best_mask >> d) & 1:
            routing[d] = 1.0

    return routing

# EVOLVE-BLOCK-END
