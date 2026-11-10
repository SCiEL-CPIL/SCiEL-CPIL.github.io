# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    """
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill

    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    n = 10  # number of nodes/edges
    D = len(demands)

    # Precompute: for each demand, which edges are used clockwise and counter-clockwise
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod 10)
    cw_edges = []  # list of lists of edge indices for clockwise
    ccw_edges = []  # list of lists of edge indices for counter-clockwise

    for s, t, w in demands:
        # Clockwise edges
        cw = []
        edge = s
        while True:
            cw.append(edge)
            if edge == t - 1 if t > s else (t - 1 + n):
                break
            edge = (edge + 1) % n
        cw_edges.append(cw)
        # Counter-clockwise: all other edges
        ccw = [e for e in range(n) if e not in set(cw)]
        ccw_edges.append(ccw)

    # Brute force all 2^15 = 32768 combinations
    best_max_load = float('inf')
    best_routing = None

    for mask in range(1 << D):
        # Check deadline
        if time.time() - start_time > deadline:
            break

        loads = np.zeros(n)
        for d in range(D):
            s, t, w = demands[d]
            if (mask >> d) & 1:
                # counter-clockwise
                for e in ccw_edges[d]:
                    loads[e] += w
            else:
                # clockwise
                for e in cw_edges[d]:
                    loads[e] += w

        max_load = np.max(loads)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = np.array([(mask >> d) & 1 for d in range(D)], dtype=np.float64)

    if best_routing is None:
        best_routing = np.zeros(D)

    return best_routing


# EVOLVE-BLOCK-END
