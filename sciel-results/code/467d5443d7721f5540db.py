# EVOLVE-BLOCK-START
import time
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Uses exhaustive search over 2^15 = 32768 possible routings, which is fast enough
    to find the global optimum well within the time limit.

    Each entry of the returned array selects the direction for one demand:
        value < 0.5 -> route the demand clockwise
        value >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one value per demand.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin for 630s limit

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute edge incidence matrices
    # cw_mask[d, e] = 1 if demand d uses edge e when routed clockwise
    # ccw_mask[d, e] = 1 if demand d uses edge e when routed counter-clockwise
    cw_mask = np.zeros((D, n), dtype=np.int8)
    ccw_mask = np.zeros((D, n), dtype=np.int8)
    
    for d, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            cw_mask[d, e] = 1
            e = (e + 1) % n
        # Counter-clockwise uses all other edges
        ccw_mask[d, :] = 1 - cw_mask[d, :]

    weights = np.array([w for (_, _, w) in demands], dtype=np.float64)
    
    # Precompute base loads for all-clockwise and all-counter-clockwise
    # base_cw_load[e] = sum of weights of demands that use edge e when routed clockwise
    # base_ccw_load[e] = sum of weights of demands that use edge e when routed counter-clockwise
    base_cw_load = weights @ cw_mask  # shape (n,)
    base_ccw_load = weights @ ccw_mask  # shape (n,)
    
    # delta_load[d, e] = change in load on edge e when switching demand d from clockwise to counter-clockwise
    # If demand d switches from cw to ccw:
    #   edges in cw_mask[d] lose weight w_d
    #   edges in ccw_mask[d] gain weight w_d
    # So delta_load[d, e] = w_d * (ccw_mask[d, e] - cw_mask[d, e])
    delta_load = weights[:, np.newaxis] * (ccw_mask.astype(np.float64) - cw_mask.astype(np.float64))  # shape (D, n)
    
    best_max_load = float('inf')
    best_routing = np.zeros(D)
    
    # Exhaustive search over all 2^D combinations
    for mask in range(1 << D):
        # Check deadline
        if time.time() > deadline - 10.0:
            break
        
        # Convert mask to array of 0s and 1s
        # bit d of mask = 1 means demand d goes counter-clockwise
        x = np.array([(mask >> d) & 1 for d in range(D)], dtype=np.float64)
        
        # Compute loads: load[e] = base_cw_load[e] + sum_d x_d * delta_load[d, e]
        loads = base_cw_load + delta_load.T @ x
        
        max_load = np.max(loads)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = x
    
    return best_routing

# EVOLVE-BLOCK-END
