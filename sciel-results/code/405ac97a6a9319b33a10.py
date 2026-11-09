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
        < 0.5 -> route the demand clockwise
        >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    D = 15
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute clockwise and counter-clockwise edge sets for each demand
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = []
        e = s
        while e != t:
            cw.append(e)
            e = (e + 1) % n
        cw_edges.append(cw)
        ccw = [e for e in range(n) if e not in cw]
        ccw_edges.append(ccw)

    # Build incidence matrix: A[d, e] = 1 if demand d uses edge e when routed CW
    A = np.zeros((D, n))
    for d in range(D):
        for e in cw_edges[d]:
            A[d, e] = 1

    # Weights vector
    w = np.array([demands[d][2] for d in range(D)], dtype=float)

    # Total weight
    total_w = w.sum()

    # For each demand, if routed CW, it adds w[d] to edges in cw_edges[d]
    # If routed CCW, it adds w[d] to edges in ccw_edges[d]
    # Let x[d] = 1 if CCW, 0 if CW
    # Load on edge e = sum_d w[d] * (A[d,e] * (1-x[d]) + (1-A[d,e]) * x[d])
    #               = sum_d w[d] * A[d,e] + sum_d w[d] * (1 - 2*A[d,e]) * x[d]

    base_load = A.T @ w  # load on each edge if all routed CW
    delta = (1 - 2 * A) * w  # change in load on edge e when demand d is switched to CCW
    # Load(e) = base_load[e] + sum_d delta[d, e] * x[d]
    #         = base_load[e] + (delta.T @ x)[e]

    # Objective: minimize max_e Load(e)
    # This is a small MILP with 15 binary variables. We can brute force 2^15 = 32768 solutions.
    # But let's try a smarter approach first.

    # Brute force is fast enough for 2^15 = 32768
    best_load = np.inf
    best_x = None

    # Use a heuristic: start with all CW, then try flipping demands that reduce the max load
    x = np.zeros(D)
    current_load = base_load.copy()
    max_load = current_load.max()

    # Local search: iteratively flip the demand that most reduces the max edge load
    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_d = -1
        for d in range(D):
            # Try flipping demand d
            new_load = current_load + delta[d]
            new_max = new_load.max()
            if new_max < max_load - 1e-9:
                best_delta = max_load - new_max
                best_d = d
        if best_d >= 0:
            x[best_d] = 1 - x[best_d]
            current_load = current_load + delta[best_d]
            max_load = current_load.max()
            improved = True

    best_load = max_load
    best_x = x.copy()

    # Try all single flips from the current solution
    for d in range(D):
        new_load = current_load + delta[d]
        new_max = new_load.max()
        if new_max < best_load - 1e-9:
            best_load = new_max
            best_x = x.copy()
            best_x[d] = 1 - best_x[d]

    # Try all pairs of flips (15*14/2 = 105 pairs)
    for d1 in range(D):
        for d2 in range(d1 + 1, D):
            new_load = current_load + delta[d1] + delta[d2]
            new_max = new_load.max()
            if new_max < best_load - 1e-9:
                best_load = new_max
                best_x = x.copy()
                best_x[d1] = 1 - best_x[d1]
                best_x[d2] = 1 - best_x[d2]

    # Try all triples (15*14*13/6 = 455 triples)
    for d1 in range(D):
        for d2 in range(d1 + 1, D):
            for d3 in range(d2 + 1, D):
                new_load = current_load + delta[d1] + delta[d2] + delta[d3]
                new_max = new_load.max()
                if new_max < best_load - 1e-9:
                    best_load = new_max
                    best_x = x.copy()
                    best_x[d1] = 1 - best_x[d1]
                    best_x[d2] = 1 - best_x[d2]
                    best_x[d3] = 1 - best_x[d3]

    # Return the routing as 0/1 array (0 = CW, 1 = CCW)
    return best_x.astype(float)

# EVOLVE-BLOCK-END
