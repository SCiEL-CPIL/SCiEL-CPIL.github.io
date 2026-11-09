import numpy as np
import time

def ring_routing() -> np.ndarray:
    n_nodes = 10
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
        (5, 2, 2),
    ]
    
    D = len(demands)
    
    # Precompute CW and CCW edge masks
    W_cw = np.zeros((D, n_nodes))
    W_ccw = np.zeros((D, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise edges: s, s+1, ..., t-1 (mod n_nodes)
        edge = s
        while edge != t:
            W_cw[i, edge] = w
            edge = (edge + 1) % n_nodes
        
        # CCW edges are the complement
        for e in range(n_nodes):
            if W_cw[i, e] == 0:
                W_ccw[i, e] = w
    
    # L = L_base + sum(x_i * Delta_i) where x_i is 1 if CCW, 0 if CW
    L_base = np.sum(W_cw, axis=0)
    Delta = W_ccw - W_cw
    
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    # Precompute all binary vectors for speed
    # Use Gray code to minimize changes between consecutive masks, but for 2^15 it's fast enough
    for mask in range(1 << D):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Convert mask to binary vector
        x = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.float64)
        
        # Calculate loads: L = L_base + x^T @ Delta (broadcast)
        loads = L_base + np.dot(x, Delta)
        
        max_load = np.max(loads)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = x.copy()
    
    return best_routing
