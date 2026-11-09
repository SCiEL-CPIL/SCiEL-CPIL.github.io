import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    # Define demands
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute CW and CCW edge usage
    # cw_edge[i, e] is 1 if demand i uses edge e when routed CW, 0 otherwise
    # ccw_edge[i, e] is 1 if demand i uses edge e when routed CCW, 0 otherwise
    
    cw_usage = np.zeros((n_demands, n_nodes), dtype=float)
    ccw_usage = np.zeros((n_demands, n_nodes), dtype=float)
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        # Edge e connects node e and (e+1)%10
        # CW from s to t uses edges s, s+1, ..., t-1
        if s <= t:
            edges_cw = range(s, t)
        else:
            edges_cw = list(range(s, n_nodes)) + list(range(0, t))
            
        for e in edges_cw:
            cw_usage[i, e] = 1.0
            
        # CCW uses all other edges
        # Total edges = 10. If CW uses k edges, CCW uses 10-k edges.
        # Actually, it's easier to just iterate all edges and check if not in CW.
        # Or simply: ccw_usage[i, :] = 1 - cw_usage[i, :]
        # Wait, does a demand use ALL non-CW edges?
        # Yes, "counter-clockwise uses all other ring edges".
        # The ring has 10 edges. The path from s to t in one direction is a subset of edges.
        # The other path is the complement.
        # So yes, ccw_usage[i, :] = 1.0 - cw_usage[i, :]
        
        ccw_usage[i, :] = 1.0 - cw_usage[i, :]
        
    # Weights
    weights = np.array([d[2] for d in demands])
    
    # Load matrices
    # load_cw[e] = sum(w_i * cw_usage[i, e])
    # load_ccw[e] = sum(w_i * ccw_usage[i, e])
    # But we need the linear combination for the LP.
    # L_e = sum_i (1-x_i) * w_i * cw_usage[i, e] + x_i * w_i * ccw_usage[i, e]
    # L_e = sum_i w_i * cw_usage[i, e] + x_i * w_i * (ccw_usage[i, e] - cw_usage[i, e])
    
    base_load = np.sum(weights[:, np.newaxis] * cw_usage, axis=0) # shape (10,)
    delta_load = weights[:, np.newaxis] * (ccw_usage - cw_usage) # shape (15, 10)
    
    # LP Variables: x_0..x_14 (binary), L (continuous)
    # z = [x_0, ..., x_14, L]
    n_vars = n_demands + 1
    idx_L = n_demands
    
    # Objective: minimize L
    c = np.zeros(n_vars)
    c[idx_L] = 1.0
    
    # Integrality
    integrality = np.zeros(n_vars)
    integrality[:n_demands] = 1.0 # x_i are integers (binary)
    integrality[idx_L] = 0.0 # L is continuous
    
    # Bounds
    # x_i in [0, 1]
    # L in [0, inf]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[idx_L] = np.inf
    
    # Constraints:
    # For each edge e:
    # sum_i x_i * delta_load[i, e] - L <= -base_load[e]
    # A matrix shape (10, 16)
    A = np.zeros((n_nodes, n_vars))
    b = np.zeros(n_nodes)
    
    for e in range(n_nodes):
        # Coefficients for x_i
        A[e, :n_demands] = delta_load[:, e]
        # Coefficient for L
        A[e, idx_L] = -1.0
        # RHS
        b[e] = -base_load[e]
        
    constraints = LinearConstraint(A, -np.inf, b)
    bounds = Bounds(lb, ub)
    
    # Solve
    res = milp(c, integrality=integrality, constraints=constraints, bounds=bounds)
    
    if res.status == 0:
        x = res.x[:n_demands]
        # Round to be safe, though milp should return integers
        x = np.round(x).astype(int)
        return x.astype(float)
    else:
        # Fallback to all clockwise if solver fails (should not happen)
        return np.zeros(n_demands)
