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
        value < 0.5 -> route the demand clockwise
        value >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one value per demand.
    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s limit

    n_nodes = 10
    D = 15

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
        (5, 2, 2)
    ]

    # Precompute CW and CCW edge weight matrices
    # cw_weights[i, e] = weight of demand i on edge e if routed CW, else 0
    # ccw_weights[i, e] = weight of demand i on edge e if routed CCW, else 0
    cw_weights = np.zeros((D, n_nodes), dtype=np.float64)
    ccw_weights = np.zeros((D, n_nodes), dtype=np.float64)

    for i, (s, t, w) in enumerate(demands):
        # Clockwise edges: s, s+1, ..., t-1 (mod n)
        node = s
        while node != t:
            cw_weights[i, node] = w
            node = (node + 1) % n_nodes
        
        # Counter-clockwise edges: all other edges
        ccw_weights[i, :] = w - cw_weights[i, :]

    # base_load[e] = sum of cw_weights[i, e] for all i
    # This is the load if all demands are routed CW.
    base_load = np.sum(cw_weights, axis=0)
    
    # delta_load[i, e] = ccw_weights[i, e] - cw_weights[i, e]
    # If demand i is routed CCW (bit 1), we add delta_load[i, e] to the load.
    # If demand i is routed CW (bit 0), we add 0 (since base_load already includes cw_weights[i, e]).
    # Load[e] = base_load[e] + sum_{i: bit_i=1} delta_load[i, e]
    delta_load = ccw_weights - cw_weights

    # Brute force all 2^15 = 32768 combinations
    # This is fast enough to complete in well under a second
    best_max_load = float('inf')
    best_combo = 0

    for combo in range(1 << D):
        # Calculate current loads
        current_load = base_load.copy()
        
        # Add contributions from CCW routed demands
        c = combo
        for i in range(D):
            if c & 1:
                current_load += delta_load[i]
            c >>= 1
        
        max_load = np.max(current_load)
        if max_load < best_max_load:
            best_max_load = max_load
            best_combo = combo
    
    # Construct the routing array
    routing = np.zeros(D)
    for i in range(D):
        if (best_combo >> i) & 1:
            routing[i] = 1.0  # CCW
        else:
            routing[i] = 0.0  # CW

    return routing


# EVOLVE-BLOCK-END
