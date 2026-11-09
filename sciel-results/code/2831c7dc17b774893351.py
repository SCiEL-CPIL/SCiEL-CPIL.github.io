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
        # Counter-clockwise: all edges not in clockwise
        all_edges = set(range(n))
        ccw = list(all_edges - set(cw))
        ccw_edges.append(ccw)

    def compute_max_load(routing):
        """Compute the maximum edge load given a routing vector."""
        loads = np.zeros(n)
        for i in range(D):
            s, t, w = demands[i]
            if routing[i] < 0.5:  # clockwise
                for e in cw_edges[i]:
                    loads[e] += w
            else:  # counter-clockwise
                for e in ccw_edges[i]:
                    loads[e] += w
        return loads.max()

    best_max_load = float('inf')
    best_routing = np.zeros(D)
    
    # Exhaustive search: 2^15 = 32768 combinations
    for mask in range(1 << D):
        routing = np.array([(mask >> i) & 1 for i in range(D)], dtype=float)
        current_max = compute_max_load(routing)
        if current_max < best_max_load:
            best_max_load = current_max
            best_routing = routing.copy()

    return best_routing


# EVOLVE-BLOCK-END
