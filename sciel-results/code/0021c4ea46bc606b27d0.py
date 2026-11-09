import numpy as np

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0,5,3), (1,7,2), (2,9,4), (3,6,1), (0,4,5), 
        (2,8,2), (5,9,3), (1,4,2), (6,0,4), (3,8,1), 
        (7,2,3), (4,9,2), (8,1,5), (0,6,1), (5,2,2)
    ]
    
    # Precompute edge masks for CW and CCW for each demand
    # cw_mask[i] is a list of edges used by demand i if routed CW
    # ccw_mask[i] is a list of edges used by demand i if routed CCW
    cw_edges = []
    ccw_edges = []
    weights = []
    
    for s, t, w in demands:
        weights.append(w)
        
        # CW path: s -> s+1 -> ... -> t
        cw_path = []
        e = s
        while e != t:
            cw_path.append(e)
            e = (e + 1) % N
        cw_edges.append(cw_path)
        
        # CCW path: uses all edges NOT in CW path
        # Equivalently, the path from t -> s in CW direction
        ccw_path = []
        e = t
        while e != s:
            ccw_path.append(e)
            e = (e + 1) % N
        ccw_edges.append(ccw_path)
    
    weights_np = np.array(weights, dtype=np.float64)
    
    # Precompute the contribution of each demand to each edge for CW and CCW
    # cw_contrib[i, e] = weight_i if edge e is in CW path of demand i, else 0
    # ccw_contrib[i, e] = weight_i if edge e is in CCW path of demand i, else 0
    cw_contrib = np.zeros((D, N), dtype=np.float64)
    ccw_contrib = np.zeros((D, N), dtype=np.float64)
    
    for i in range(D):
        w_i = weights[i]
        for e in cw_edges[i]:
            cw_contrib[i, e] = w_i
        for e in ccw_edges[i]:
            ccw_contrib[i, e] = w_i
    
    best_max_load = float('inf')
    best_mask = 0
    
    # Precompute delta and base load
    delta_contrib = ccw_contrib - cw_contrib
    base_load = np.sum(cw_contrib, axis=0)
    
    # Brute force all 2^15 combinations
    for mask in range(1 << D):
        bits = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.float64)
        loads = base_load + bits @ delta_contrib
        current_max = np.max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = mask

    # Construct result
    routing = np.zeros(D)
    for i in range(D):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
