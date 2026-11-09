# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
        0 -> clockwise, 1 -> counter-clockwise
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
        # Clockwise edges: s, (s+1)%n, ..., (t-1)%n
        cw = []
        if s <= t:
            for e in range(s, t):
                cw.append(e)
        else:
            for e in range(s, n):
                cw.append(e)
            for e in range(0, t):
                cw.append(e)
        cw_edges.append(cw)

        # Counter-clockwise edges: all edges not in clockwise
        all_edges = list(range(n))
        ccw = [e for e in all_edges if e not in cw]
        ccw_edges.append(ccw)

    weights = np.array([d[2] for d in demands], dtype=np.float64)

    # Precompute contribution vectors: for each demand, the edge load vector if routed CW or CCW
    cw_contrib = np.zeros((D, n), dtype=np.float64)
    ccw_contrib = np.zeros((D, n), dtype=np.float64)

    for d in range(D):
        w = weights[d]
        for e in cw_edges[d]:
            cw_contrib[d, e] = w
        for e in ccw_edges[d]:
            ccw_contrib[d, e] = w

    # Precompute: for each demand, the delta vector (change in edge loads) when flipping from CW to CCW
    # delta[d] = ccw_contrib[d] - cw_contrib[d]
    delta = ccw_contrib - cw_contrib  # shape (D, n)

    # Base load: all demands routed clockwise
    base_load = np.sum(cw_contrib, axis=0)  # shape (n,)

    # Brute force all 2^15 = 32768 combinations using vectorized delta updates
    best_max_load = float('inf')
    best_mask = 0

    start_time = time.time()
    deadline = start_time + 570.0  # safety margin

    for mask in range(1 << D):
        if time.time() > deadline:
            break

        # Compute edge loads: base_load + sum of delta[d] for each demand d that is CCW (bit set)
        # Vectorized: extract bits, multiply by delta rows, sum
        # For speed with 15 bits, we can build the load incrementally or use a precomputed approach
        
        # Efficient approach: precompute all possible masks? No, 32768 is fine with fast inner loop
        # Use numpy: create a boolean vector of which demands are CCW
        ccw_mask = np.array([(mask >> d) & 1 for d in range(D)], dtype=np.float64)
        # loads = base_load + ccw_mask @ delta
        loads = base_load + ccw_mask @ delta
        max_load = np.max(loads)

        if max_load < best_max_load:
            best_max_load = max_load
            best_mask = mask

    # Build the routing array from best_mask
    routing = np.zeros(D, dtype=np.float64)
    for d in range(D):
        if (best_mask >> d) & 1:
            routing[d] = 1.0

    return routing

# EVOLVE-BLOCK-END
