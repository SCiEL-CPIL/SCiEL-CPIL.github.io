import numpy as np
import time
from scipy.optimize import linprog

def ring_routing() -> np.ndarray:
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    cw_vec = np.zeros((n_demands, n_nodes), dtype=np.int8)
    ccw_vec = np.zeros((n_demands, n_nodes), dtype=np.int8)
    
    for i, (s, t, w) in enumerate(demands):
        current = s
        while current != t:
            cw_vec[i, current] = 1
            current = (current + 1) % n_nodes
        ccw_vec[i] = 1 - cw_vec[i]
        
    weights = np.array([w for s, t, w in demands], dtype=np.int32)
    diff = ccw_vec - cw_vec
    
    # Formulate as LP:
    # Variables: x_0, ..., x_{14} (direction, 0=CW, 1=CCW), x_15 = L (max load)
    # Objective: minimize L
    # Constraints: For each edge e, sum_i w_i * (cw_i_e + x_i * diff_i_e) <= L
    #              x_i in [0, 1], L >= 0
    
    # Edge load = sum_i w_i * cw_i_e + sum_i w_i * diff_i_e * x_i
    # For each edge e: sum_i w_i * diff_i_e * x_i - L <= -sum_i w_i * cw_i_e
    
    # A matrix: n_nodes rows, (n_demands + 1) columns
    # First n_demands columns: w_i * diff_i_e for demand i, edge e
    # Last column: -1 for L term
    
    A = np.zeros((n_nodes, n_demands + 1))
    b = np.zeros(n_nodes)
    
    for e in range(n_nodes):
        for i in range(n_demands):
            A[e, i] = weights[i] * diff[i, e]
        A[e, n_demands] = -1.0
        b[e] = -np.sum(weights * cw_vec[:, e])
    
    # Objective: minimize L -> c vector
    c = np.zeros(n_demands + 1)
    c[n_demands] = 1.0
    
    # Bounds: x_i in [0,1], L >= 0
    bounds = [(0, 1)] * n_demands + [(0, None)]
    
    # Solve LP
    start_time = time.time()
    deadline = start_time + 580.0
    
    try:
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if result.success:
            x = result.x[:n_demands]
            # Round to 0 or 1
            routing = (x >= 0.5).astype(np.float64)
            return routing
    except Exception:
        pass
    
    # Fallback to exhaustive search if LP fails or times out
    if time.time() > deadline:
        return np.zeros(n_demands, dtype=np.float64)
    
    # Vectorized exhaustive search
    n_masks = 1 << n_demands
    masks = np.arange(n_masks, dtype=np.uint16)
    directions = ((masks[:, None] >> np.arange(n_demands)) & 1).astype(np.int8)
    
    used = cw_vec[None, :, :] + directions[:, :, None] * diff[None, :, :]
    loads = np.sum(used * weights[None, :, None], axis=1)
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_routing = directions[best_idx].astype(np.float64)
    
    return best_routing
