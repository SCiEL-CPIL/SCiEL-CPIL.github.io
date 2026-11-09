import numpy as np

def ring_routing() -> np.ndarray:
    n_nodes = 10
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
        (5, 2, 2),
    ]
    D = len(demands)
    
    # base_loads[e] is the load on edge e if all demands are routed Clockwise
    base_loads = np.zeros(n_nodes, dtype=np.float64)
    # diff_loads[i][e] is the change in load on edge e if demand i is switched from CW to CCW
    diff_loads = np.zeros((D, n_nodes), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        cw_edges = set()
        e = s
        while e != t:
            cw_edges.add(e)
            e = (e + 1) % n_nodes
        
        # Add to base loads (CW contribution)
        for e in cw_edges:
            base_loads[e] += w
            
        # Calculate difference (CCW - CW)
        # CCW edges are all edges not in CW edges
        for e in range(n_nodes):
            if e in cw_edges:
                # If routed CW: contributes w. If routed CCW: contributes 0.
                # So switching CW->CCW changes load by -w
                diff_loads[i, e] = -w
            else:
                # If routed CW: contributes 0. If routed CCW: contributes w.
                # So switching CW->CCW changes load by +w
                diff_loads[i, e] = w
                
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=np.float64)
    
    # Use Gray Code to iterate through all 2^D states
    # current_loads starts as base_loads (all CW, routing = 0)
    current_loads = base_loads.copy()
    current_routing = np.zeros(D, dtype=np.float64)
    
    for i in range(1 << D):
        # Calculate max load for current state
        max_load = np.max(current_loads)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = current_routing.copy()
            
        # If not the last state, calculate next Gray code state
        if i < (1 << D) - 1:
            # Find the bit that changes in the next Gray code
            # Gray code g(i) = i ^ (i >> 1)
            # g(i+1) ^ g(i) gives the bit that flips
            g_i = i ^ (i >> 1)
            g_next = (i + 1) ^ ((i + 1) >> 1)
            changed_bit = g_i ^ g_next
            
            # Index of the bit that changed (0-indexed from LSB)
            bit_idx = (changed_bit.bit_length() - 1)
            
            if (g_next >> bit_idx) & 1:
                # Bit changed from 0 to 1: Switch demand bit_idx from CW to CCW
                current_loads += diff_loads[bit_idx]
                current_routing[bit_idx] = 1.0
            else:
                # Bit changed from 1 to 0: Switch demand bit_idx from CCW to CW
                current_loads -= diff_loads[bit_idx]
                current_routing[bit_idx] = 0.0
                
    return best_routing
