import numpy as np
import time

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute CW and CCW edge masks
    # cw_mask[i][e] is 1 if edge e is used by demand i in CW, 0 otherwise
    # Actually, we need the weight.
    # Let's use arrays of length N for each demand.
    
    cw_weights = np.zeros((D, N), dtype=np.float64)
    ccw_weights = np.zeros((D, N), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        steps_cw = (t - s) % N
        if steps_cw > 0:
            for k in range(steps_cw):
                e = (s + k) % N
                cw_weights[i, e] = w
        
        steps_ccw = (s - t) % N
        if steps_ccw > 0:
            for k in range(steps_ccw):
                e = (s - 1 - k) % N
                ccw_weights[i, e] = w
                
    # Base load is sum of all CW weights
    base_load = np.sum(cw_weights, axis=0)
    
    # Delta is ccw - cw
    delta = ccw_weights - cw_weights
    
    best_load = float('inf')
    best_mask = 0
    
    # Exhaustive search
    # 2^15 = 32768 masks
    for mask in range(1 << D):
        # Calculate load
        # load = base_load + sum( delta[i] for i where bit i is set )
        
        # Optimization: We can iterate through bits.
        # But N=10 is small.
        
        # Using numpy might be faster if we vectorize the mask selection?
        # But the mask changes.
        
        # Let's stick to the efficient loop or a numpy trick.
        # Numpy trick:
        # bits = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.float64)
        # load = base_load + bits @ delta
        # This creates an array of 15 and does a matmul (15x10).
        # This is very fast in C.
        
        bits = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.float64)
        load = base_load + bits.dot(delta)
        
        current_max = np.max(load)
        
        if current_max < best_load:
            best_load = current_max
            best_mask = mask
            
    # Construct result
    result = np.zeros(D, dtype=np.float64)
    for i in range(D):
        if (best_mask >> i) & 1:
            result[i] = 1.0 # CCW
        else:
            result[i] = 0.0 # CW
            
    return result
