import numpy as np
import time

def ring_routing() -> np.ndarray:
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
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute the contribution of each demand to each edge for both CW and CCW routes.
    cw_contrib = np.zeros((n_demands, n_nodes))
    ccw_contrib = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        if s == t:
            continue
        
        # Determine CW edges
        if s < t:
            edges = list(range(s, t))
        else:
            edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in edges:
            cw_contrib[i, e] = w
            
        # Determine CCW edges (complement of CW edges)
        cw_edges_set = set(edges)
        for e in range(n_nodes):
            if e not in cw_edges_set:
                ccw_contrib[i, e] = w

    # Base load if all demands were routed CW
    base_load = np.sum(cw_contrib, axis=0)
    
    # Delta load for each demand if we switch from CW to CCW
    deltas = ccw_contrib - cw_contrib
    
    # We iterate through all 2^15 masks to find the routing that minimizes the max edge load.
    best_max_load = float('inf')
    best_mask = 0
    
    start_time = time.time()
    timeout = 630.0
    safety_margin = 10.0
    
    # Optimization: Use Gray code to update loads incrementally.
    # Gray code ensures only one bit changes between consecutive masks.
    
    # We'll iterate through all masks in Gray code order.
    # For mask m, the Gray code is m ^ (m >> 1).
    # To go from gray(m-1) to gray(m), we flip the bit at position ctz(m) (count trailing zeros).
    
    current_load = base_load.copy()
    current_max = np.max(current_load)
    
    if current_max < best_max_load:
        best_max_load = current_max
        best_mask = 0
        
    for m in range(1, 1 << n_demands):
        if (time.time() - start_time) > (timeout - safety_margin):
            break
            
        # Find the bit that changes from gray(m-1) to gray(m)
        # This is the position of the least significant set bit in m
        # ctz(m) = (m & -m).bit_length() - 1
        bit_to_flip = (m & -m).bit_length() - 1
        
        # Determine if the bit is being set or cleared
        # In Gray code, if bit is set in m but not in m-1, it's being set.
        # Actually, simpler: just check the bit in the current gray code mask?
        # No, we need to know the direction.
        # Let's use the property: the bit that flips is the one that changes.
        # If we are going from gray(m-1) to gray(m), the bit at position ctz(m) flips.
        # We need to know if it flips 0->1 or 1->0.
        # We can track the current mask state, or just check the parity?
        # Actually, we can just apply the delta. If the bit was 0, we add delta. If 1, we subtract.
        # We can determine the current state of that bit in the *previous* gray code mask.
        # But we don't store the mask.
        # Alternative: Just recompute the load from scratch? No, that defeats the purpose.
        # Let's track the current mask state.
        
        # Wait, the standard trick is:
        # gray(m) = m ^ (m >> 1)
        # The bit that changes between gray(m-1) and gray(m) is at index ctz(m).
        # To know if we add or subtract, we can check if that bit is 1 in gray(m).
        # If it is 1, it means it was 0 in gray(m-1) and is now 1 (so we ADD delta).
        # If it is 0, it means it was 1 in gray(m-1) and is now 0 (so we SUBTRACT delta).
        
        gray_m = m ^ (m >> 1)
        if (gray_m >> bit_to_flip) & 1:
            current_load += deltas[bit_to_flip]
        else:
            current_load -= deltas[bit_to_flip]
            
        current_max = np.max(current_load)
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = gray_m

    # Construct result array
    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
