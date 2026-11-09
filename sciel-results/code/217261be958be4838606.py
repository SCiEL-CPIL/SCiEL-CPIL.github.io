import numpy as np
import time

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
    
    # Brute force all 2^15 combinations
    for mask in range(1 << D):
        # For each demand i, if bit i is 1, use CCW, else CW
        # We can compute the load as:
        # load = sum_i ( (bit_i ? ccw_contrib[i] : cw_contrib[i]) )
        
        # Efficient way:
        # Start with sum of all CW contributions
        # For each bit set, add (ccw_contrib[i] - cw_contrib[i])
        
        # Precompute delta_contrib = ccw_contrib - cw_contrib
        # load = base_load + sum_{i in mask} delta_contrib[i]
        
        # This is much faster than matrix multiplication per iteration if we update incrementally,
        # but since we iterate over all masks, we can use a Gray code or just simple vectorization.
        
        # Given D=15, simple vectorization is fast enough.
        # Let's stick to the previous method which was ~0.1s.
        
        # Alternative: Precompute delta
        # delta = ccw_contrib - cw_contrib (15x10)
        # base_load = sum(cw_contrib, axis=0) (10,)
        
        # For a given mask, loads = base_load + sum_{i: bit_i=1} delta[i]
        
        # We can precompute the cumulative sum for all masks? No, 32768 * 10 floats is small.
        # But calculating sum for each mask from scratch is O(D*N) = 150 ops.
        # 32768 * 150 = ~5 million ops. Very fast.
        
        pass

    # Let's implement the delta method for elegance and speed
    delta_contrib = ccw_contrib - cw_contrib
    base_load = np.sum(cw_contrib, axis=0)
    
    for mask in range(1 << D):
        # Calculate loads
        # We need sum of delta_contrib[i] for all i where bit i is set.
        # This can be done with a dot product of the mask bits and delta_contrib.
        
        # Create bits array? That was the bottleneck in the previous thought process, but it's tiny.
        # bits = np.array([(mask >> i) & 1 for i in range(D)])
        # loads = base_load + bits @ delta_contrib
        
        # Even faster: iterate through bits?
        # No, numpy is fine.
        
        # To avoid allocating `bits` array every time, we can use a lookup table or just keep it.
        # 32768 allocations of size 15 is negligible.
        
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
