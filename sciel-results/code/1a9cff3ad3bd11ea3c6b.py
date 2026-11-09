# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    D = 15
    N = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        cw_mask = [0.0] * N
        ccw_mask = [0.0] * N
        
        steps_cw = (t - s) % N
        if steps_cw > 0:
            for i in range(steps_cw):
                e = (s + i) % N
                cw_mask[e] = w
        
        steps_ccw = (s - t) % N
        if steps_ccw > 0:
            for i in range(steps_ccw):
                e = (s - 1 - i) % N
                ccw_mask[e] = w
                
        cw_edges.append(cw_mask)
        ccw_edges.append(ccw_mask)
    
    base_load = [0.0] * N
    delta = []
    for i in range(D):
        for e in range(N):
            base_load[e] += cw_edges[i][e]
        d = [ccw_edges[i][e] - cw_edges[i][e] for e in range(N)]
        delta.append(d)
    
    best_load = float('inf')
    best_routing = [0.0] * D
    
    start_time = time.time()
    deadline = start_time + 550.0
    
    for mask in range(1 << D):
        if time.time() > deadline:
            break
        
        loads = base_load[:]
        for i in range(D):
            if (mask >> i) & 1:
                d = delta[i]
                for e in range(N):
                    loads[e] += d[e]
        
        current_max_load = max(loads)
        
        if current_max_load < best_load:
            best_load = current_max_load
            best_routing = [(1.0 if (mask >> i) & 1 else 0.0) for i in range(D)]
            
    return np.array(best_routing, dtype=np.float64)

# EVOLVE-BLOCK-END
