import numpy as np
import time

def ring_routing() -> np.ndarray:
    t_start = time.time()
    # 600s limit, use 590s as safety margin for return
    deadline = t_start + 590.0
    
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
    cw_loads = np.zeros((n_demands, n_nodes))
    ccw_loads = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # CW path from s to t: edges s, s+1, ..., t-1 (mod 10)
        if s == t:
            # No edges used? Or full circle? 
            # "CW from s to t uses edges s, ..., t-1". If s==t, range is empty?
            # Usually s!=t in demands. Here all s!=t.
            pass
        elif s < t:
            edges = np.arange(s, t)
        else:
            edges = np.concatenate([np.arange(s, n_nodes), np.arange(0, t)])
        
        cw_loads[i, edges] = w
        
        # CCW uses all other edges
        # Mask of edges NOT in CW
        cw_mask = np.zeros(n_nodes, dtype=bool)
        cw_mask[edges] = True
        ccw_mask = ~cw_mask
        ccw_loads[i, ccw_mask] = w
        
    # Base load if all demands are routed CW
    base = np.sum(cw_loads, axis=0)
    # Delta if demand i is switched from CW to CCW
    deltas = ccw_loads - cw_loads
    
    # Exhaustive search over 2^15 masks
    num_masks = 1 << n_demands
    masks = np.arange(num_masks)
    
    # Extract bits: shape (num_masks, n_demands)
    # masks[:, None] >> np.arange(n_demands)
    bits = ((masks[:, None] >> np.arange(n_demands)[None, :]) & 1).astype(np.float64)
    
    # Compute loads: (num_masks, n_nodes)
    loads = base[None, :] + (bits @ deltas)
    
    # Find mask with minimum max load
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = int(masks[best_idx])
    best_max_load = float(max_loads[best_idx])
    
    # Construct result
    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0  # CCW
        else:
            routing[i] = 0.0  # CW
            
    return routing
