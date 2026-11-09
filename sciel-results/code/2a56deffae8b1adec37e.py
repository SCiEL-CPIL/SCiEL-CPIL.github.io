import numpy as np
import time

def ring_routing() -> np.ndarray:
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]

    # Precompute edge usage matrices
    # cw_mat[i, e] = 1 if edge e is used by demand i in CW direction
    # ccw_mat[i, e] = 1 if edge e is used by demand i in CCW direction
    cw_mat = np.zeros((D, n), dtype=np.float64)
    ccw_mat = np.zeros((D, n), dtype=np.float64)
    weights = np.zeros(D, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        d_cw = (t - s) % n
        if d_cw == 0:
            # Same node, no edges used in either direction for load purposes in this context?
            # Problem statement: "Clockwise from s to t uses edges s, ..., t-1".
            # If s==t, range is empty.
            pass
        else:
            # CW edges: s, s+1, ..., t-1
            for k in range(d_cw):
                e = (s + k) % n
                cw_mat[i, e] = 1.0
            # CCW edges: all other edges
            for e in range(n):
                if cw_mat[i, e] == 0.0:
                    ccw_mat[i, e] = 1.0

    # Base load if all demands go CW
    base_load = weights @ cw_mat # shape (n,)
    
    # Delta vector for flipping demand i from CW to CCW
    # delta[i] = w_i * (ccw_mat[i] - cw_mat[i])
    delta_mat = (ccw_mat - cw_mat) * weights[:, None]
    
    # Exhaustive search over all 2^15 routing combinations
    # This is guaranteed to find the global optimum for this problem size.
    best_max_load = np.inf
    best_mask = 0
    
    start_time = time.time()
    deadline = start_time + 590.0 # Safety margin
    
    for mask in range(1 << D):
        if time.time() > deadline:
            break
            
        # Calculate load for this mask
        # load = base_load + sum(delta_mat[i] for i where bit i is set)
        load = base_load.copy()
        for i in range(D):
            if mask & (1 << i):
                load += delta_mat[i]
        
        max_load = np.max(load)
        if max_load < best_max_load - 1e-9:
            best_max_load = max_load
            best_mask = mask

    # Reconstruct the routing vector from the best mask
    # Entry >= 0.5 routes CCW (1), else CW (0)
    out = np.zeros(D, dtype=np.float64)
    for i in range(D):
        if best_mask & (1 << i):
            out[i] = 1.0
        else:
            out[i] = 0.0
            
    return out

# The function is defined, no need to call it here as the evaluator will call it.
# However, the seed program had a call at the bottom. I will omit it to avoid side effects 
# unless required, but standard practice for these tasks is just to define the function.
# The prompt says "Keep the function names and signatures of the seed program."
# It does not say to keep the top-level call. The evaluator calls ring_routing().
