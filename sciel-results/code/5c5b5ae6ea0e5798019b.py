import numpy as np
import time

def ring_routing() -> np.ndarray:
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Pre-compute for each demand: which edges are used in CW and CCW
    cw_mask = np.zeros((n_demands, n_nodes), dtype=np.float64)
    ccw_mask = np.zeros((n_demands, n_nodes), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        current = s
        while current != t:
            cw_mask[i, current] = 1.0
            current = (current + 1) % n_nodes
        ccw_mask[i] = 1.0 - cw_mask[i]
    
    weights = np.array([w for _, _, w in demands])
    cw_weighted = cw_mask * weights[:, np.newaxis]
    ccw_weighted = ccw_mask * weights[:, np.newaxis]
    
    base_load = cw_weighted.sum(axis=0)
    delta = ccw_weighted - cw_weighted
    
    best_max_load = float('inf')
    best_mask = 0
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Gray code iteration for O(1) update per step
    loads = base_load.copy()
    
    # Check mask=0 (all CW)
    current_max = base_load.max()
    best_max_load = current_max
    best_mask = 0
    
    for k in range(1, 1 << n_demands):
        if time.time() > deadline:
            break
        
        lsb = k & (-k)
        bit_pos = lsb.bit_length() - 1
        new_gray = k ^ (k >> 1)
        
        if (new_gray >> bit_pos) & 1:
            loads += delta[bit_pos]
        else:
            loads -= delta[bit_pos]
        
        current_max = loads.max()
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = new_gray
    
    # Local improvement: try flipping individual bits to see if we can reduce max load
    improved = True
    while improved:
        improved = False
        for i in range(n_demands):
            # Try flipping bit i
            if (best_mask >> i) & 1:
                # Currently CCW, try CW
                test_loads = loads - delta[i]
            else:
                # Currently CW, try CCW
                test_loads = loads + delta[i]
            
            test_max = test_loads.max()
            
            if test_max < best_max_load:
                best_max_load = test_max
                best_mask ^= (1 << i)
                loads = test_loads
                improved = True
    
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
    
    return result
