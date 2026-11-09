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
    
    # Precompute load contributions for each demand in CW and CCW
    # cw_load[i] is a list of 10 ints, load on each edge
    # ccw_load[i] is a list of 10 ints, load on each edge
    
    cw_loads = []
    ccw_loads = []
    
    for s, t, w in demands:
        cw_edges = []
        ccw_edges = []
        
        # Determine CW edges
        # Path s -> t clockwise
        # If s < t: s, s+1, ..., t-1
        # If s > t: s, s+1, ..., 9, 0, ..., t-1
        # If s == t: no edges? (Not in this problem)
        
        if s < t:
            cw_edges = list(range(s, t))
        elif s > t:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        else:
            cw_edges = []
            
        # CCW edges are the complement
        all_edges = set(range(n_nodes))
        cw_set = set(cw_edges)
        ccw_edges = list(all_edges - cw_set)
        
        # Create load vectors
        cw_vec = [0] * n_nodes
        for e in cw_edges:
            cw_vec[e] = w
            
        ccw_vec = [0] * n_nodes
        for e in ccw_edges:
            ccw_vec[e] = w
            
        cw_loads.append(cw_vec)
        ccw_loads.append(ccw_vec)
        
    best_max_load = float('inf')
    best_mask = 0
    
    # Total combinations: 2^15 = 32768
    # We can iterate through all masks
    # To speed up, we can use a recursive approach or just a loop.
    # A loop is simple and fast enough.
    
    # Optimization: 
    # Instead of recalculating the sum from scratch for each mask,
    # we can iterate through masks and use Gray code to flip one bit at a time,
    # updating the loads incrementally.
    # However, 32768 * 15 * 10 operations is small.
    # Let's try the simple loop first.
    
    # Even better: Precompute the total load for each edge as a function of the mask?
    # No, let's just loop.
    
    # To make it even faster, we can use numpy to vectorize the mask evaluation?
    # Creating an array of 32768 masks is easy.
    # But summing 15 vectors for each of 32768 masks...
    # Let's stick to Python loops for simplicity and reliability, it's fast enough.
    
    # Actually, let's use a slightly optimized loop.
    # We can precompute the delta between CCW and CW for each demand.
    # load = base_load + sum(deltas where bit is set)
    # base_load = sum(cw_loads[i] for all i)
    # delta[i] = ccw_loads[i] - cw_loads[i]
    
    base_load = [0] * n_nodes
    for i in range(n_demands):
        for e in range(n_nodes):
            base_load[e] += cw_loads[i][e]
            
    deltas = []
    for i in range(n_demands):
        d = [0] * n_nodes
        for e in range(n_nodes):
            d[e] = ccw_loads[i][e] - cw_loads[i][e]
        deltas.append(d)
        
    # Now for each mask, load = base_load + sum(deltas[i] for i where bit i is set)
    # This is still a sum over set bits.
    
    # Let's just do the direct summation, it's clearer.
    
    start_time = time.time()
    timeout = 600.0
    safety_margin = 10.0
    
    # We can break early if we find a perfect solution? 
    # The minimum possible max load is at least max(weight)/2? No.
    # The sum of weights is 3+2+4+1+5+2+3+2+4+1+3+2+5+1+2 = 40.
    # Total load on all edges = sum of weights * (length of path)? No.
    # Each demand adds weight to every edge on its path.
    # The sum of loads on all 10 edges is not constant.
    
    # Let's just iterate.
    
    for mask in range(1 << n_demands):
        # Check time
        if (time.time() - start_time) > (timeout - safety_margin):
            break
            
        loads = [0] * n_nodes
        for i in range(n_demands):
            if (mask >> i) & 1:
                # CCW
                vec = ccw_loads[i]
            else:
                # CW
                vec = cw_loads[i]
            # Add vec to loads
            for e in range(n_nodes):
                loads[e] += vec[e]
                
        current_max = max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = mask
            
    # Construct result array
    # Entry >= 0.5 means CCW (1), < 0.5 means CW (0)
    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
