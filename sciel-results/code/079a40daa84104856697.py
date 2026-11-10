import numpy as np

def ring_routing() -> np.ndarray:
    n_nodes = 10
    n_demands = 15
    
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
    
    # Precompute edge indices for CW and CCW for each demand
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        # Calculate CW edges
        dist = (t - s) % n_nodes
        if dist == 0:
            # s == t, loop around? 
            # "Clockwise from s to t uses edges s, s+1, ..., t-1"
            # If s==t, distance is 0. Usually implies 0 edges or full loop?
            # In ring routing, s!=t is standard. Let's assume standard distinct nodes.
            # All demands have s != t.
            pass
        cw_set = set()
        for k in range(dist):
            cw_set.add((s + k) % n_nodes)
        
        cw_list = list(cw_set)
        ccw_list = [e for e in range(n_nodes) if e not in cw_set]
        
        cw_edges.append(cw_list)
        ccw_edges.append(ccw_list)
    
    best_max_load = float('inf')
    best_mask = 0
    
    # Brute force all 2^15 combinations
    for mask in range(1 << n_demands):
        loads = [0] * n_nodes
        
        for i in range(n_demands):
            weight = demands[i][2]
            if (mask >> i) & 1:
                # CCW
                for e in ccw_edges[i]:
                    loads[e] += weight
            else:
                # CW
                for e in cw_edges[i]:
                    loads[e] += weight
        
        current_max = max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = mask
            
    # Construct result
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            result[i] = 1.0  # >= 0.5 means CCW
        else:
            result[i] = 0.0  # < 0.5 means CW
            
    return result
