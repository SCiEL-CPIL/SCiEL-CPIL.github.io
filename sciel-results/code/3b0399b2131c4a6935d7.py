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
        (5, 2, 2)
    ]
    
    num_demands = len(demands)
    
    # Precompute edges for each demand for both directions
    # clockwise_edges[i] is a list of edge indices for demand i if routed clockwise
    # counter_clockwise_edges[i] is a list of edge indices for demand i if routed counter-clockwise
    clockwise_edges = []
    counter_clockwise_edges = []
    
    for s, t, w in demands:
        cw_edges = []
        curr = s
        while curr != t:
            cw_edges.append(curr)
            curr = (curr + 1) % n_nodes
        
        # Counter-clockwise uses all other edges
        ccw_edges = [e for e in range(n_nodes) if e not in cw_edges]
        
        clockwise_edges.append(cw_edges)
        counter_clockwise_edges.append(ccw_edges)
        
    best_max_load = float('inf')
    best_routing = np.zeros(num_demands)
    
    # Exhaustive search over all 2^15 routing combinations
    for mask in range(1 << num_demands):
        loads = np.zeros(n_nodes)
        
        for i in range(num_demands):
            w = demands[i][2]
            if (mask >> i) & 1:
                # Counter-clockwise
                for e in counter_clockwise_edges[i]:
                    loads[e] += w
            else:
                # Clockwise
                for e in clockwise_edges[i]:
                    loads[e] += w
        
        current_max_load = np.max(loads)
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            # Construct the routing array
            # 1.0 for CCW (bit set), 0.0 for CW (bit not set)
            for i in range(num_demands):
                if (mask >> i) & 1:
                    best_routing[i] = 1.0
                else:
                    best_routing[i] = 0.0
                    
    return best_routing
