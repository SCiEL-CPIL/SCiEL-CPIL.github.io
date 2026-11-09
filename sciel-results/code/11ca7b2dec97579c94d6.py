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
    
    deadline = time.time() + 600.0
    
    # Precompute edge lists
    cw_edges = []
    ccw_edges = []
    weights = []
    
    for s, t, w in demands:
        weights.append(w)
        # CW: s -> t
        edges = []
        e = s
        while e != t:
            edges.append(e)
            e = (e + 1) % N
        cw_edges.append(edges)
        
        # CCW: t -> s
        edges = []
        e = t
        while e != s:
            edges.append(e)
            e = (e + 1) % N
        ccw_edges.append(edges)
    
    # Use numpy arrays for faster load computation
    cw_arr = np.zeros((D, N), dtype=np.float64)
    ccw_arr = np.zeros((D, N), dtype=np.float64)
    for i in range(D):
        for e in cw_edges[i]:
            cw_arr[i, e] = 1.0
        for e in ccw_edges[i]:
            ccw_arr[i, e] = 1.0
    w_arr = np.array(weights, dtype=np.float64)
    
    best_max_load = float('inf')
    best_assignment = None
    
    # Brute force 2^15 = 32768 combinations
    for mask in range(1 << D):
        if time.time() > deadline - 10.0:
            break
        
        # Compute loads: for each demand, pick CW or CCW based on bit
        # loads = sum of (bit_i * ccw_arr[i] + (1-bit_i) * cw_arr[i]) * w_i
        bits = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.float64)
        # For each demand i: if bit=1 use ccw, else use cw
        route_arr = bits[:, None] * ccw_arr + (1.0 - bits[:, None]) * cw_arr
        loads = route_arr.T @ w_arr
        
        current_max = np.max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_assignment = mask
            
    # Convert best_assignment to output array
    # Entry >= 0.5 -> CCW (1), < 0.5 -> CW (0)
    routing = np.zeros(D)
    for i in range(D):
        if (best_assignment >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
