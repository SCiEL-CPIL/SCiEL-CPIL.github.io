import numpy as np

def ring_routing() -> np.ndarray:
    D = 15
    demands = [
        (0,5,3), (1,7,2), (2,9,4), (3,6,1), (0,4,5), (2,8,2), (5,9,3), (1,4,2), 
        (6,0,4), (3,8,1), (7,2,3), (4,9,2), (8,1,5), (0,6,1), (5,2,2)
    ]
    n = 10
    
    # Precompute edge indices for CW and CCW
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = []
        ccw = []
        cur = s
        while cur != t:
            cw.append(cur)
            cur = (cur + 1) % n
        cw_edges.append(cw)
        
        cur = s
        while cur != t:
            ccw.append((cur - 1) % n)
            cur = (cur - 1) % n
        ccw_edges.append(ccw)
        
    best_max_load = float('inf')
    best_routing = None
    
    # Brute force search
    for mask in range(1 << D):
        loads = [0.0] * n
        for i in range(D):
            w = demands[i][2]
            if (mask >> i) & 1:
                for e in ccw_edges[i]:
                    loads[e] += w
            else:
                for e in cw_edges[i]:
                    loads[e] += w
        
        max_load = max(loads)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = np.array([1.0 if (mask >> i) & 1 else 0.0 for i in range(D)])
            
    return best_routing
