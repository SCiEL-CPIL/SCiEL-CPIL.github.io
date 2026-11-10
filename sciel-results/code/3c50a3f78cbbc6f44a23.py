import numpy as np
import time

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge masks
    # cw_mask[i, e] = 1 if edge e is in CW path for demand i
    # ccw_mask[i, e] = 1 if edge e is in CCW path for demand i
    cw_masks = np.zeros((D, n), dtype=np.float64)
    ccw_masks = np.zeros((D, n), dtype=np.float64)
    weights = np.zeros(D, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        if s == t:
            # Degenerate case, but not present here
            continue
        e = s
        while e != t:
            cw_masks[i, e] = 1.0
            e = (e + 1) % n
        ccw_masks[i] = 1.0 - cw_masks[i]
        
    # Precompute delta: change in load if we switch from CW to CCW
    # delta = ccw_mask - cw_mask
    # If state is 0 (CW) -> 1 (CCW): loads += weight * delta
    # If state is 1 (CCW) -> 0 (CW): loads -= weight * delta
    delta_masks = ccw_masks - cw_masks
    weighted_deltas = delta_masks * weights[:, np.newaxis]
    
    # Initialize loads for all CW (state = 0)
    # load_e = sum_{i: e in cw_i} w_i
    # This is equivalent to (cw_masks.T @ weights)
    loads = cw_masks.T @ weights
    
    best_max_load = np.max(loads)
    best_routing = np.zeros(D, dtype=np.float64)
    
    # Gray code enumeration
    # g = i ^ (i >> 1)
    # diff = g ^ g_prev
    # bit = diff.bit_length() - 1
    
    deadline = time.time() + 590
    total_states = 1 << D
    
    state = np.zeros(D, dtype=np.int8) # 0 for CW, 1 for CCW
    
    for i in range(1, total_states):
        if time.time() > deadline:
            break
            
        g = i ^ (i >> 1)
        g_prev = (i - 1) ^ ((i - 1) >> 1)
        diff = g ^ g_prev
        bit = diff.bit_length() - 1
        
        # Determine direction of change
        # If state[bit] was 0, it becomes 1 (CW -> CCW)
        # If state[bit] was 1, it becomes 0 (CCW -> CW)
        
        if state[bit] == 0:
            loads += weighted_deltas[bit]
            state[bit] = 1
        else:
            loads -= weighted_deltas[bit]
            state[bit] = 0
            
        current_max = np.max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_routing = state.astype(np.float64).copy()
            
    return best_routing
