# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Exhaustive search over all 2^15 routing assignments to find the one that
    minimizes the maximum edge load on a 10-node ring with 15 demands.
    """
    N = 10
    D = 15
    start_time = time.time()

    # Demands: (source, target, weight)
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

    # Precompute for each demand:
    #   cw_edges[d]: set of edges used if routed clockwise
    #   ccw_edges[d]: set of edges used if routed counter-clockwise
    # Edge e connects node e to node (e+1) % N.
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod N)
    # Counter-clockwise: all other edges

    cw_edge_mask = np.zeros((D, N), dtype=np.int8)  # 1 if edge used in CW
    ccw_edge_mask = np.zeros((D, N), dtype=np.int8)  # 1 if edge used in CCW

    for d, (s, t, w) in enumerate(demands):
        if s == t:
            # No edges needed (shouldn't happen here)
            continue
        # Clockwise edges: s, s+1, ..., t-1 (mod N)
        e = s
        while True:
            cw_edge_mask[d, e] = 1
            if e == (t - 1) % N:
                break
            e = (e + 1) % N
        # Counter-clockwise: all edges NOT in clockwise
        ccw_edge_mask[d] = 1 - cw_edge_mask[d]

    weights = np.array([w for (_, _, w) in demands], dtype=np.float64)

    # Exhaustive search over all 2^D assignments
    # For each assignment x in {0,1}^D (0=cw, 1=ccw):
    #   load[e] = sum_d weights[d] * (x[d] ? ccw_edge_mask[d,e] : cw_edge_mask[d,e])
    #   = sum_d weights[d] * (cw_edge_mask[d,e] + x[d] * (ccw_edge_mask[d,e] - cw_edge_mask[d,e]))

    # Precompute:
    # base_load[e] = sum_d weights[d] * cw_edge_mask[d,e]
    # delta_mask[d,e] = ccw_edge_mask[d,e] - cw_edge_mask[d,e]  (in {-1, 0, 1})
    base_load = weights @ cw_edge_mask  # shape (N,)
    delta_mask = (ccw_edge_mask - cw_edge_mask).astype(np.float64)  # shape (D, N)

    # For each assignment x (D-bit vector), load = base_load + x @ delta_mask
    # We want to minimize max(load)

    # Enumerate all 2^D assignments
    total = 1 << D  # 32768
    best_max_load = float('inf')
    best_x = None

    # Use numpy to vectorize over chunks of assignments
    # For each assignment x, compute load = base_load + x @ delta_mask, then max(load)
    # We can process all 32768 at once or in chunks.

    # Generate all 2^D bit vectors as a (total, D) matrix
    # bits[i] = binary representation of i, with bit d being the d-th bit
    all_x = np.zeros((total, D), dtype=np.float64)
    for d in range(D):
        # bit d of i is (i >> d) & 1
        all_x[:, d] = (np.arange(total) >> d) & 1

    # Compute loads for all assignments at once: shape (total, N)
    # load[i, e] = base_load[e] + sum_d all_x[i, d] * delta_mask[d, e]
    loads = base_load[np.newaxis, :] + all_x @ delta_mask  # (total, N)

    # Max load per assignment
    max_loads = loads.max(axis=1)  # (total,)

    # Find the assignment with minimum max load
    idx = np.argmin(max_loads)
    best_max_load = max_loads[idx]
    best_x = all_x[idx]

    # Convert best_x (0/1 array) to routing: >= 0.5 means counter-clockwise
    routing = best_x.copy()

    return routing


# EVOLVE-BLOCK-END
