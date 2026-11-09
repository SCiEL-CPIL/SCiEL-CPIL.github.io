# EVOLVE-BLOCK-START
import numpy as np


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    D = 15
    N = 10  # number of nodes/edges
    
    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute for each demand: which edges are used in CW and CCW
    # cw_edges[i] is a bitmask of 10 bits indicating which edges are used when demand i is routed CW
    # ccw_edges[i] is a bitmask of 10 bits indicating which edges are used when demand i is routed CCW
    cw_mask = np.zeros(D, dtype=np.int32)
    ccw_mask = np.zeros(D, dtype=np.int32)
    weights = np.zeros(D, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        # CW from s to t: edges s, s+1, ..., t-1 (mod 10)
        cw = 0
        e = s
        while e != t:
            cw |= (1 << e)
            e = (e + 1) % N
        cw_mask[i] = cw
        # CCW: all edges not in CW
        ccw_mask[i] = ((1 << N) - 1) ^ cw
    
    # Exhaustive search over all 2^D routings
    # For each routing (bitmask r), bit i = 0 means CW, bit i = 1 means CCW
    best_max_load = float('inf')
    best_routing = 0
    
    for r in range(1 << D):
        # Compute edge loads
        edge_loads = np.zeros(N, dtype=np.float64)
        for i in range(D):
            if (r >> i) & 1:
                # CCW
                mask = ccw_mask[i]
                for e in range(N):
                    if (mask >> e) & 1:
                        edge_loads[e] += weights[i]
            else:
                # CW
                mask = cw_mask[i]
                for e in range(N):
                    if (mask >> e) & 1:
                        edge_loads[e] += weights[i]
        
        ml = np.max(edge_loads)
        if ml < best_max_load:
            best_max_load = ml
            best_routing = r
    
    # Convert best_routing bitmask to output array
    routing = np.zeros(D, dtype=np.float64)
    for i in range(D):
        if (best_routing >> i) & 1:
            routing[i] = 1.0  # CCW
        else:
            routing[i] = 0.0  # CW
    
    return routing


# EVOLVE-BLOCK-END
