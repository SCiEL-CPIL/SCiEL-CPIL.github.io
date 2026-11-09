import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute incidence vectors
    # cw[e, i] is 1 if edge e is on clockwise path of demand i
    cw = np.zeros((n, D), dtype=np.int8)
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            edges = range(s, t)
        else:
            edges = list(range(s, n)) + list(range(0, t))
        for e in edges:
            cw[e, i] = 1
            
    ccw = 1 - cw
    
    # Weights vector
    weights = np.array([d[2] for d in demands], dtype=float)
    
    # Coefficients for x_i in the constraint for edge e
    # Load_e = sum_i w_i * cw[e,i] * (1-x_i) + w_i * ccw[e,i] * x_i
    # Load_e = sum_i w_i * cw[e,i] + x_i * sum_i w_i * (ccw[e,i] - cw[e,i])
    # Constraint: sum_i w_i * (ccw[e,i] - cw[e,i]) * x_i - M <= - sum_i w_i * cw[e,i]
    
    # Matrix A for x variables (size n x D)
    # A[e, i] = w_i * (ccw[e,i] - cw[e,i])
    diff = ccw - cw
    A_x = (diff.T * weights).T  # shape (n, D)
    
    # RHS for constraints (size n)
    # b_ub[e] = - sum_i w_i * cw[e,i]
    b_ub = -np.sum(cw.T * weights, axis=1)
    
    # Full A matrix (n x (D+1))
    A = np.zeros((n, D + 1))
    A[:, :D] = A_x
    A[:, D] = -1.0
    
    # Objective
    c = np.zeros(D + 1)
    c[D] = 1.0
    
    # Bounds
    lb = np.zeros(D + 1)
    ub = np.ones(D + 1)
    ub[D] = np.inf
    
    # Integrality
    integrality = np.ones(D + 1)
    integrality[D] = 0
    
    res = milp(
        c=c,
        constraints=[LinearConstraint(A, ub=b_ub)],
        integrality=integrality,
        bounds=Bounds(lb, ub)
    )
    
    if res.status == 0:
        x = res.x[:D]
        routing = np.where(x > 0.5, 1.0, 0.0)
        return routing
    else:
        # Fallback: Greedy or random? 
        # Since 2^15 is small, we can do a quick local search or just return zeros.
        # But zeros is bad. Let's try a simple heuristic: 
        # Start with all 0, flip bits that reduce max load.
        routing = np.zeros(D)
        best_load = np.inf
        
        # Calculate load for a given routing
        def calc_load(r):
            load = np.zeros(n)
            for i in range(D):
                if r[i] < 0.5: # CW
                    load += weights[i] * cw[:, i]
                else: # CCW
                    load += weights[i] * ccw[:, i]
            return np.max(load)
            
        best_load = calc_load(routing)
        
        # Simple local search
        improved = True
        while improved:
            improved = False
            for i in range(D):
                old_val = routing[i]
                routing[i] = 1.0 - old_val
                new_load = calc_load(routing)
                if new_load < best_load:
                    best_load = new_load
                    improved = True
                else:
                    routing[i] = old_val
        return routing
