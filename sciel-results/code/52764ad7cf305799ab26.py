import numpy as np
import time
from scipy.optimize import linprog

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem using a min-max linear programming formulation.
    Variables: x_i in [0, 1] for each demand i (fraction routed CCW).
    Objective: minimize L (max edge load).
    Constraints: For each edge e, sum_i (load_cw[i,e] * (1-x_i) + load_ccw[i,e] * x_i) <= L.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # Demand definitions: (source, target, weight)
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

    n_nodes = 10
    n_demands = len(demands)

    # Pre-compute load vectors for clockwise and counter-clockwise routing
    cw_loads = np.zeros((n_demands, n_nodes))
    ccw_loads = np.zeros((n_demands, n_nodes))

    for i, (s, t, w) in enumerate(demands):
        # Clockwise: edges s, s+1, ..., t-1 (mod n_nodes)
        current = s
        while current != t:
            cw_loads[i, current] += w
            current = (current + 1) % n_nodes

        # Counter-clockwise: all other edges
        for e in range(n_nodes):
            if cw_loads[i, e] == 0:
                ccw_loads[i, e] = w

    # Solve the LP to get the optimal fractional solution
    # Variables: x_0, ..., x_14 (15 vars), L (1 var). Total 16 vars.
    # x_i in [0, 1]
    # L >= 0
    # For each edge e (10 constraints):
    # sum_i (cw_loads[i,e] + (ccw_loads[i,e] - cw_loads[i,e])*x_i) <= L
    # => sum_i ((ccw_loads[i,e] - cw_loads[i,e])*x_i) - L <= -sum_i cw_loads[i,e]

    n_vars = n_demands + 1
    c = np.zeros(n_vars)
    c[-1] = 1.0  # Minimize L

    A_ub = np.zeros((n_nodes, n_vars))
    b_ub = np.zeros(n_nodes)

    for e in range(n_nodes):
        for i in range(n_demands):
            A_ub[e, i] = ccw_loads[i, e] - cw_loads[i, e]
        A_ub[e, -1] = -1.0
        b_ub[e] = -cw_loads[:, e].sum()

    bounds = [(0, 1)] * n_demands + [(0, None)]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0 and res.x is not None:
            x_lp = res.x[:n_demands]
            # Round LP solution to binary routing decisions
            # If x_i >= 0.5, route CCW (1.0), else CW (0.0)
            routing = np.where(x_lp >= 0.5, 1.0, 0.0)

            # Verify the solution
            loads = np.zeros(n_nodes)
            for i in range(n_demands):
                if routing[i] >= 0.5:
                    loads += ccw_loads[i]
                else:
                    loads += cw_loads[i]

            max_load = np.max(loads)

            # If the rounded solution is good enough, return it
            # Otherwise, fall back to exhaustive search
            if max_load <= 6:  # Reasonable threshold
                return routing
    except Exception:
        pass

    # Fallback: vectorized exhaustive search
    # Precompute: for each mask, load = sum(cw_loads) + sum(delta_loads[i] for i where bit i is set)
    # where delta_loads[i] = ccw_loads[i] - cw_loads[i]
    delta_loads = ccw_loads - cw_loads  # shape (n_demands, n_nodes)
    base_load = cw_loads.sum(axis=0)  # shape (n_nodes,)
    
    best_max_load = float('inf')
    best_routing = np.zeros(n_demands)
    best_mask = 0
    
    # Process in chunks to use numpy efficiently
    chunk_size = 4096
    total_masks = 1 << n_demands
    for start in range(0, total_masks, chunk_size):
        if time.time() > deadline:
            break
        end = min(start + chunk_size, total_masks)
        masks = np.arange(start, end)
        
        # For each mask, compute which bits are set
        # Create a boolean matrix: (chunk_size, n_demands)
        bits = ((masks[:, None] >> np.arange(n_demands)) & 1).astype(float)
        
        # Compute loads: base_load + bits @ delta_loads
        loads = base_load[None, :] + bits @ delta_loads  # shape (chunk_size, n_nodes)
        
        # Find max load per mask
        max_loads = loads.max(axis=1)
        
        # Find best in this chunk
        local_best_idx = np.argmin(max_loads)
        local_best_load = max_loads[local_best_idx]
        
        if local_best_load < best_max_load:
            best_max_load = local_best_load
            best_mask = masks[local_best_idx]
            best_routing = ((best_mask >> np.arange(n_demands)) & 1).astype(float)
            
            if best_max_load <= 4:
                break
    
    return best_routing
