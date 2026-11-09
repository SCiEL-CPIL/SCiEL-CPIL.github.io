import numpy as np
import time

def ring_routing() -> np.ndarray:
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Build masks
    # cw_mask[i, e] is 1 if edge e is in CW path for demand i
    cw_mask = np.zeros((D, n_nodes), dtype=int)
    ccw_mask = np.zeros((D, n_nodes), dtype=int)
    
    for i, (s, t, w) in enumerate(demands):
        if s == t:
            # CW is empty, CCW is all
            ccw_mask[i, :] = 1
        else:
            e = s
            while e != t:
                cw_mask[i, e] = 1
                e = (e + 1) % n_nodes
            # CCW is the rest
            ccw_mask[i, :] = 1 - cw_mask[i, :]
            
    # Weights vector
    weights = np.array([d[2] for d in demands], dtype=float)
    
    # Load if all CW
    # L_cw[e] = sum(weights[i] * cw_mask[i, e])
    L_cw = weights @ cw_mask
    
    # Change in load if we switch demand i to CCW
    # Delta[i, e] = weights[i] * (ccw_mask[i, e] - cw_mask[i, e])
    Delta = (weights[:, None] * (ccw_mask - cw_mask))
    
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 590 # Safety margin
    
    # Precompute bit vectors for all masks? No, 32k * 15 floats is small.
    # But creating the array inside the loop is slow.
    # We can iterate and update.
    
    # Optimization: Use integer arithmetic for loads if possible?
    # Weights are integers. Masks are integers.
    # L_cw is integer. Delta is integer.
    # We can do this in integers to avoid float overhead?
    # But np.max on int array is fine.
    
    L_cw_int = L_cw.astype(int)
    Delta_int = Delta.astype(int)
    
    for mask in range(1 << D):
        if time.time() > deadline:
            break
            
        # Extract bits
        # This list comprehension is the bottleneck.
        # Alternative: use a precomputed table of bits for 0..32767?
        # Or just use bit shifting.
        
        # Fast bit extraction using numpy?
        # x = np.array([(mask >> i) & 1 for i in range(D)], dtype=int)
        
        # Let's try to compute loads incrementally?
        # No, random access in exhaustive search is hard.
        
        # Just use the float version but with @
        x = np.array([(mask >> i) & 1 for i in range(D)], dtype=float)
        
        loads = L_cw + (x @ Delta)
        
        max_load = np.max(loads)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = x
    
    return best_routing
