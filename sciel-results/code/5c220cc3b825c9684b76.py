import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

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

    # Precompute CW masks
    cw_masks = np.zeros((D, n_nodes), dtype=int)
    
    for i, (s, t, w) in enumerate(demands):
        edge = s
        while edge != t:
            cw_masks[i, edge] = 1
            edge = (edge + 1) % n_nodes

    weights = np.array([d[2] for d in demands], dtype=int)
    
    # Variables: x_0, ..., x_{D-1} (binary, 1=CCW, 0=CW), M (continuous)
    # Total variables: D + 1
    n_vars = D + 1
    
    # Objective: minimize M
    c = np.zeros(n_vars)
    c[D] = 1.0  # M is the last variable
    
    # Constraints: for each edge e, load_e <= M
    # load_e = sum_i w_i * (cw_masks[i,e] * (1-x_i) + (1-cw_masks[i,e]) * x_i)
    #        = sum_i w_i * cw_masks[i,e] + sum_i w_i * x_i * (1 - 2*cw_masks[i,e])
    # So: sum_i w_i * (1 - 2*cw_masks[i,e]) * x_i - M <= -sum_i w_i * cw_masks[i,e]
    
    A = np.zeros((n_nodes, n_vars))
    b_lower = np.zeros(n_nodes)
    b_upper = np.zeros(n_nodes)
    
    for e in range(n_nodes):
        for i in range(D):
            A[e, i] = weights[i] * (1 - 2 * cw_masks[i, e])
        A[e, D] = -1.0  # coefficient of M
        b_upper[e] = -np.sum(weights * cw_masks[:, e])
    
    # x_i in {0, 1}
    integrality = np.ones(n_vars)
    integrality[D] = 0  # M is continuous
    
    constraints = LinearConstraint(A, b_lower, b_upper)
    bounds = Bounds(0, 1, n_vars)
    
    try:
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if result.status == 0 and result.x is not None:
            x = result.x[:D]
            routing = np.where(x > 0.5, 1.0, 0.0)
            
            # Verify by computing actual max load
            ccw_masks = 1 - cw_masks
            load = np.zeros(n_nodes)
            for i in range(D):
                if routing[i] > 0.5:
                    load += weights[i] * ccw_masks[i]
                else:
                    load += weights[i] * cw_masks[i]
            max_load = np.max(load)
            
            # Also do a quick exhaustive check to verify optimality
            best_max_load = float('inf')
            best_routing = np.zeros(D)
            for mask in range(1 << D):
                if time.time() > deadline:
                    break
                load_check = np.zeros(n_nodes)
                for i in range(D):
                    if (mask >> i) & 1:
                        load_check += weights[i] * ccw_masks[i]
                    else:
                        load_check += weights[i] * cw_masks[i]
                ml = np.max(load_check)
                if ml < best_max_load:
                    best_max_load = ml
                    best_routing = np.zeros(D)
                    for j in range(D):
                        if (mask >> j) & 1:
                            best_routing[j] = 1.0
            
            # Return the better of ILP solution and exhaustive
            if max_load <= best_max_load:
                return routing
            else:
                return best_routing
        else:
            raise ValueError("ILP failed")
    except Exception:
        pass
    
    # Fallback: exhaustive search
    ccw_masks = 1 - cw_masks
    best_max_load = float('inf')
    best_routing = np.zeros(D)
    for mask in range(1 << D):
        if time.time() > deadline:
            break
        load = np.zeros(n_nodes)
        for i in range(D):
            if (mask >> i) & 1:
                load += weights[i] * ccw_masks[i]
            else:
                load += weights[i] * cw_masks[i]
        ml = np.max(load)
        if ml < best_max_load:
            best_max_load = ml
            best_routing = np.zeros(D)
            for j in range(D):
                if (mask >> j) & 1:
                    best_routing[j] = 1.0
    
    return best_routing
