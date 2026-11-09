import numpy as np

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute CW masks
    # cw_mask[d, e] = 1 if edge e is on the clockwise path from s to t
    cw_mask = np.zeros((D, n), dtype=np.int8)
    for d, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            cw_mask[d, e] = 1
            e = (e + 1) % n
            
    weights = np.array([d[2] for d in demands], dtype=float)
    
    # B: Load if all demands are routed CW
    # B[e] = sum_d w_d * cw_mask[d, e]
    B = (weights[:, None] * cw_mask).sum(axis=0)
    
    # D_vec[d]: Difference in load vector if demand d is switched from CW to CCW
    # Load_CCW(d) = w_d * (1 - cw_mask[d])
    # Load_CW(d) = w_d * cw_mask[d]
    # Delta = Load_CCW - Load_CW = w_d * (1 - 2 * cw_mask[d])
    # 1 is a vector of ones of length n
    ones = np.ones(n, dtype=np.int8)
    D_vec = weights[:, None] * (ones[None, :] - 2 * cw_mask)
    
    # Iterate through all 2^D subsets using Gray code
    # gray = g ^ (g >> 1)
    # We want to find the subset S of demands routed CCW
    # that minimizes max( B + sum_{d in S} D_vec[d] )
    
    import time
    deadline = time.time() + 590  # Safety margin before 600s limit
    
    best_max_load = float('inf')
    best_gray = 0
    current_sum = np.zeros(n, dtype=float)
    
    prev_gray = 0
    # We only care about the state of the bits, so we can just iterate g
    for g in range(2**D):
        if time.time() > deadline:
            break
        gray = g ^ (g >> 1)
        
        if g > 0:
            diff = gray ^ prev_gray
            idx = diff.bit_length() - 1
            
            # Check if the bit is set in the new gray code
            if (gray >> idx) & 1:
                # Bit turned on: demand idx switched to CCW
                current_sum += D_vec[idx]
            else:
                # Bit turned off: demand idx switched to CW
                current_sum -= D_vec[idx]
                
        prev_gray = gray
        
        current_load = B + current_sum
        max_load = np.max(current_load)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_gray = gray
            
    # Construct the routing vector from best_gray
    routing = np.zeros(D, dtype=float)
    for d in range(D):
        if (best_gray >> d) & 1:
            routing[d] = 1.0  # CCW
        else:
            routing[d] = 0.0  # CW
            
    return routing
