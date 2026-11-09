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
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    start_time = time.time()
    deadline = start_time + 620  # safety margin before 630s limit

    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute Base and Delta matrices
    # Base[e] = sum of weights of demands routed CW that use edge e
    # Delta[d, e] = weight if demand d uses edge e in CCW minus weight if it uses it in CW
    # Load[e] = Base[e] + sum_d b_d * Delta[d, e]
    
    Base = np.zeros(N, dtype=np.float64)
    Delta = np.zeros((D, N), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        cw_edges = set()
        edge = s
        while edge != t:
            cw_edges.add(edge)
            edge = (edge + 1) % N
        
        for e in range(N):
            in_cw = e in cw_edges
            in_ccw = not in_cw
            
            w_cw = w if in_cw else 0.0
            w_ccw = w if in_ccw else 0.0
            
            Base[e] += w_cw
            Delta[i, e] = w_ccw - w_cw

    # Try all 2^15 = 32768 combinations
    best_load = float('inf')
    best_routing = None

    total_combos = 1 << D
    # Pre-allocate b vector
    b = np.zeros(D, dtype=np.float64)
    
    for combo in range(total_combos):
        if time.time() > deadline:
            break
        
        # Set b vector
        for i in range(D):
            b[i] = 1.0 if (combo >> i) & 1 else 0.0
        
        # Compute loads
        loads = Base + b @ Delta
        max_load = np.max(loads)
        
        if max_load < best_load:
            best_load = max_load
            best_routing = b.copy()

    # Convert to the required format: >= 0.5 means CCW, < 0.5 means CW
    if best_routing is None:
        # Fallback: all CW
        result = np.zeros(D)
    else:
        result = best_routing.astype(np.float64)

    return result


# EVOLVE-BLOCK-END
