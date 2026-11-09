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
        1 -> route the demand counter-clockwise (entry >= 0.5)

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
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0 - 1.0  # 1 second safety margin

    # Precompute edge indices for clockwise and counter-clockwise paths
    # cw_edges[i] is a list of edge indices for demand i if routed clockwise
    # ccw_edges[i] is a list of edge indices for demand i if routed counter-clockwise
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        cw = []
        e = s
        while e != t:
            cw.append(e)
            e = (e + 1) % n
        cw_edges.append(cw)
        
        ccw = []
        e = s
        while e != t:
            e = (e - 1) % n
            ccw.append(e)
        ccw_edges.append(ccw)
        
    # Weights array
    weights = np.array([w for _, _, w in demands], dtype=float)

    best_max_load = float('inf')
    best_mask = 0

    # Brute force all 2^15 = 32768 combinations
    # This is fast enough to run to completion well within the deadline
    for mask in range(1 << D):
        # Check deadline (unlikely to be needed for this size, but required by directive)
        if time.time() > deadline:
            break
            
        loads = np.zeros(n, dtype=float)
        
        # Accumulate loads
        # Iterate through each demand
        for i in range(D):
            if (mask >> i) & 1:
                # Counter-clockwise
                for e in ccw_edges[i]:
                    loads[e] += weights[i]
            else:
                # Clockwise
                for e in cw_edges[i]:
                    loads[e] += weights[i]
                    
        max_load = np.max(loads)
        if max_load < best_max_load:
            best_max_load = max_load
            best_mask = mask
            
    # Construct the result array
    # 0 for clockwise, 1 for counter-clockwise
    result = np.zeros(D, dtype=float)
    for i in range(D):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
            
    return result
