import numpy as np

def ring_routing() -> np.ndarray:
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
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute CW and CCW load vectors for each demand
    cw_loads = []
    ccw_loads = []
    
    for s, t, w in demands:
        # CW path from s to t: edges s, s+1, ..., t-1 (mod 10)
        cw_edges = []
        if s == t:
            cw_edges = []
        elif s < t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        cw_set = set(cw_edges)
        ccw_edges = [e for e in range(n_nodes) if e not in cw_set]
        
        cw_vec = np.zeros(n_nodes)
        for e in cw_edges:
            cw_vec[e] = w
            
        ccw_vec = np.zeros(n_nodes)
        for e in ccw_edges:
            ccw_vec[e] = w
            
        cw_loads.append(cw_vec)
        ccw_loads.append(ccw_vec)
    
    cw_arr = np.array(cw_loads)  # shape (15, 10)
    ccw_arr = np.array(ccw_loads)  # shape (15, 10)
    
    # Base load if all demands are routed CW
    base = np.sum(cw_arr, axis=0)  # shape (10,)
    # Delta if demand i is switched from CW to CCW
    deltas = ccw_arr - cw_arr  # shape (15, 10)
    
    # Vectorized exhaustive search
    # Number of masks is 2^15 = 32768
    num_masks = 1 << n_demands
    
    # Create a matrix of bits: shape (num_masks, n_demands)
    # bits[m, i] is 1 if bit i of mask m is set
    masks = np.arange(num_masks)
    # Use broadcasting to extract bits
    # masks[:, None] has shape (num_masks, 1)
    # np.arange(n_demands)[None, :] has shape (1, n_demands)
    bits = ((masks[:, None] >> np.arange(n_demands)[None, :]) & 1).astype(np.float64)
    
    # Compute loads for all masks: shape (num_masks, n_nodes)
    # loads = base + bits @ deltas
    # base is (10,), bits is (32768, 15), deltas is (15, 10)
    # bits @ deltas is (32768, 10)
    loads = base[None, :] + (bits @ deltas)
    
    # Find the mask with the minimum maximum load
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx]
    
    # Construct result
    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0  # CCW
        else:
            routing[i] = 0.0  # CW
            
    return routing
