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
        (5, 2, 2)
    ]
    D = len(demands)
    
    # Deadline guard: 600s limit, use 590s to be safe
    start_time = time.time()
    deadline = start_time + 590.0

    cw_matrix = np.zeros((D, n_nodes))
    ccw_matrix = np.zeros((D, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in cw_edges:
            cw_matrix[i, e] = w
            
        all_edges = set(range(n_nodes))
        cw_set = set(cw_edges)
        ccw_list = [e for e in range(n_nodes) if e not in cw_set]
        for e in ccw_list:
            ccw_matrix[i, e] = w
            
    base_loads = cw_matrix.sum(axis=0)
    delta_matrix = ccw_matrix - cw_matrix
    
    # Exhaustive search is optimal for D=15
    # 2^15 = 32768 combinations. This is very fast.
    total_bits = 2 ** D
    all_bits = np.arange(total_bits, dtype=np.int32)
    bit_positions = np.arange(D)
    
    # Routing matrix: (32768, 15)
    # 1 means CCW, 0 means CW
    r_all = ((all_bits[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1).astype(np.float64)
    
    # Loads: (32768, 10)
    loads_all = r_all @ delta_matrix + base_loads[np.newaxis, :]
    
    # Find best combination
    max_loads_all = np.max(loads_all, axis=1)
    
    # Check time, though this should be done in milliseconds
    if time.time() > deadline:
        # Fallback: return all CW (index 0) or a heuristic if we ran out of time
        # But we shouldn't be here.
        return np.zeros(D)

    best_idx = np.argmin(max_loads_all)
    best_routing = r_all[best_idx].copy()
    
    return best_routing
