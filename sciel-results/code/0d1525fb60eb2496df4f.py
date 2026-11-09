import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem for 10 nodes and 15 demands.
    Returns an array of 15 floats (0.0 or 1.0) indicating the routing direction
    for each demand to minimize the maximum edge load.
    """
    start_time = time.time()
    deadline = start_time + 600.0

    N = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute clockwise route incidence matrix
    # CW_ROUTES[i, e] is 1 if demand i uses edge e when routed clockwise
    cw_routes = np.zeros((D, N), dtype=np.int8)
    weights = np.zeros(D, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        dist = (t - s) % N
        for k in range(dist):
            cw_routes[i, (s + k) % N] = 1
            
    # CCW routes are the complement
    ccw_routes = 1 - cw_routes
    
    # Generate all 2^D routing masks
    # Bit i of mask represents the routing of demand i: 0 for CW, 1 for CCW
    masks = np.arange(1 << D, dtype=np.int32)
    
    # Create a matrix of routing decisions: shape (2^D, D)
    # routing_matrix[m, i] is 1 if demand i is routed CCW, 0 if CW
    routing_matrix = np.zeros((1 << D, D), dtype=np.float64)
    for j in range(D):
        routing_matrix[:, j] = (masks >> j) & 1
        
    # Calculate loads for all combinations
    # For each combination m:
    # Load on edge e = sum_i (routing_matrix[m, i] * ccw_routes[i, e] + (1 - routing_matrix[m, i]) * cw_routes[i, e]) * weights[i]
    
    # Vectorized calculation:
    # weights_matrix shape (1, D)
    w = weights[np.newaxis, :]
    
    # ccw_weights: weight if routed CCW
    ccw_w = routing_matrix * w
    # cw_weights: weight if routed CW
    cw_w = (1 - routing_matrix) * w
    
    # Loads shape (2^D, N)
    ccw_loads = ccw_w @ ccw_routes
    cw_loads = cw_w @ cw_routes
    total_loads = ccw_loads + cw_loads
    
    # Max load for each combination
    max_loads = total_loads.max(axis=1)
    
    # Find the combination with minimum max load
    best_idx = np.argmin(max_loads)
    best_routing = routing_matrix[best_idx]
    
    # Deadline guard check (redundant for this size but required by directive)
    if time.time() > deadline - 1.0:
        # In a more complex search, we would return best_routing here as a fallback.
        # Here, the search is complete.
        pass

    return best_routing
