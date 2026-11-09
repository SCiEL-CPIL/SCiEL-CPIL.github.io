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
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590.0

    # Precompute edge masks for CW and CCW for each demand
    # cw_contrib[i, e] is the weight added to edge e if demand i is routed CW
    # ccw_contrib[i, e] is the weight added to edge e if demand i is routed CCW
    cw_contrib = np.zeros((D, N), dtype=np.float64)
    ccw_contrib = np.zeros((D, N), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        # CW path: s -> s+1 -> ... -> t
        e = s
        while e != t:
            cw_contrib[i, e] = w
            e = (e + 1) % N
        
        # CCW path: t -> t+1 -> ... -> s (which are the edges NOT in CW)
        e = t
        while e != s:
            ccw_contrib[i, e] = w
            e = (e + 1) % N

    # Base load: all demands routed CW
    base_load = np.sum(cw_contrib, axis=0)
    
    # Delta load: change in load when switching from CW to CCW
    # delta[i, e] = ccw_contrib[i, e] - cw_contrib[i, e]
    delta_contrib = ccw_contrib - cw_contrib

    best_max_load = float('inf')
    best_gray = 0
    
    # Gray code enumeration over all 2^D routings
    # This is efficient as each step only changes one demand's route
    total = 1 << D
    prev_gray = 0
    current_loads = base_load.copy()
    
    # Check initial state (all CW, gray=0)
    cur_max = float(np.max(current_loads))
    if cur_max < best_max_load:
        best_max_load = cur_max
        best_gray = 0

    for step in range(1, total):
        gray = step ^ (step >> 1)
        diff = gray ^ prev_gray
        # Find the bit that changed
        bit = (diff & -diff).bit_length() - 1
        # Determine if the bit changed from 0 to 1 (CW to CCW) or 1 to 0 (CCW to CW)
        is_ccw = (gray >> bit) & 1

        # Update loads incrementally
        if is_ccw:
            current_loads += delta_contrib[bit]
        else:
            current_loads -= delta_contrib[bit]

        prev_gray = gray
        
        cur_max = float(np.max(current_loads))
        if cur_max < best_max_load:
            best_max_load = cur_max
            best_gray = gray

        # Time check (though this loop is very fast for D=15)
        if time.time() > deadline:
            break

    # Construct result
    # Entry >= 0.5 routes CCW (bit=1), else CW (bit=0)
    routing = np.zeros(D)
    for i in range(D):
        if (best_gray >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
