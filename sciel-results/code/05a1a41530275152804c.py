import numpy as np

def ring_routing() -> np.ndarray:
    n_nodes = 10
    n_demands = 15
    
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute load vectors for each demand for CW and CCW routing
    # cw_load[i] is a vector of length 10, where edge e has weight w if e is on CW path, else 0
    # ccw_load[i] is a vector of length 10, where edge e has weight w if e is on CCW path, else 0
    
    cw_loads = np.zeros((n_demands, n_nodes))
    ccw_loads = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise path: s -> s+1 -> ... -> t
        curr = s
        while curr != t:
            cw_loads[i, curr] = w
            curr = (curr + 1) % n_nodes
        
        # Counter-clockwise path: all other edges
        # Since it's a ring, CCW edges are simply all edges not in CW path
        # However, we must be careful: if s == t, path is empty? 
        # Here s != t for all demands.
        # The CCW path uses edges t, t+1, ..., s-1 (mod 10)
        # Or simply: all edges except those in CW path.
        # Let's verify:
        # CW edges: {e | e is on path s->t}
        # CCW edges: {e | e is NOT on path s->t}
        # This is correct for a simple ring routing where paths are disjoint except for endpoints?
        # No, the paths partition the ring edges.
        # So ccw_loads[i, e] = w if e not in cw_edges else 0.
        
        # We can compute ccw by subtracting cw from total? No, total weight is not on all edges.
        # Just compute directly or use the complement.
        # Since cw_loads[i] has w on specific edges and 0 elsewhere,
        # and the set of edges is partitioned,
        # ccw_loads[i] will have w on the complement edges.
        
        # Efficient way:
        # Create a mask of edges used in CW
        # But we already filled cw_loads.
        # We can just fill ccw_loads by iterating or using the fact that 
        # sum(cw_loads[i]) + sum(ccw_loads[i]) = w * 10? No.
        # sum(cw_loads[i]) = w * len(cw_path)
        # sum(ccw_loads[i]) = w * len(ccw_path)
        
        # Let's just compute ccw explicitly to be safe and clear.
        curr = t
        while curr != s:
            ccw_loads[i, curr] = w
            curr = (curr + 1) % n_nodes

    # Base load: all demands routed CW
    base_load = np.sum(cw_loads, axis=0)
    
    # Delta load: change in load when switching from CW to CCW
    # delta[i] = ccw_loads[i] - cw_loads[i]
    delta_loads = ccw_loads - cw_loads
    
    # Exhaustive search over 2^15 masks
    # masks: 0 to 2^15 - 1
    # Bit i is 1 -> CCW, Bit i is 0 -> CW
    
    # Vectorized computation
    # Create array of all masks
    masks = np.arange(1 << n_demands, dtype=np.int32)
    
    # Extract bits for each demand
    # bits shape: (2^15, 15)
    # bits[m, i] = (mask_m >> i) & 1
    bits = ((masks[:, None] >> np.arange(n_demands, dtype=np.int32)) & 1).astype(np.float32)
    
    # Compute loads for all masks
    # loads[m, e] = base_load[e] + sum_i(bits[m, i] * delta_loads[i, e])
    # loads = base_load[None, :] + bits @ delta_loads
    loads = base_load[None, :] + bits @ delta_loads
    
    # Find max load for each mask
    max_loads = np.max(loads, axis=1)
    
    # Find the mask with minimum max load
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx]
    
    # Construct result array
    # Entry >= 0.5 -> CCW (1.0), else CW (0.0)
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
            
    return result
