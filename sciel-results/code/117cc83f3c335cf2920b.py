import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    n_nodes = 10
    n_edges = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # A[e, i] = 1 if demand i uses edge e when routed CW
    # B[e, i] = 1 if demand i uses edge e when routed CCW
    A = np.zeros((n_edges, D), dtype=int)
    B = np.zeros((n_edges, D), dtype=int)
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        cw_edges = set()
        curr = s
        while curr != t:
            cw_edges.add(curr)
            curr = (curr + 1) % n_nodes
        
        for e in range(n_edges):
            if e in cw_edges:
                A[e, i] = 1
            else:
                B[e, i] = 1
        
    # MIP Setup
    # Variables: x[0..14] (binary), M (continuous)
    # Total vars = D + 1
    
    # Objective: Minimize M
    # c = [0, 0, ..., 0, 1]
    c = np.zeros(D + 1)
    c[D] = 1.0
    
    # Integrality: x are integer (binary), M is continuous
    integrality = np.array([1] * D + [0])
    
    # Bounds: x in [0, 1], M in [0, inf]
    lb = np.zeros(D + 1)
    ub = np.ones(D + 1)
    ub[D] = np.inf
    bounds = Bounds(lb, ub)
    
    # Constraints: L_e <= M for each edge e
    # L_e = sum_i w_i * (x_i * B[e,i] + (1-x_i) * A[e,i])
    # L_e = sum_i w_i * A[e,i] + sum_i w_i * x_i * (B[e,i] - A[e,i])
    # sum_i w_i * x_i * (B[e,i] - A[e,i]) - M <= - sum_i w_i * A[e,i]
    
    constraints = []
    weights = np.array([d[2] for d in demands])
    
    for e in range(n_edges):
        # Coefficients for x_i: w_i * (B[e,i] - A[e,i])
        coeff_x = weights * (B[e, i] - A[e, i]) # Wait, index i is not defined in loop scope properly for matrix op
        # Correct:
        diff = B[e, :] - A[e, :]
        coeff_x = weights * diff
        
        # Coefficient for M: -1
        coeff_M = -1.0
        
        # Row for constraint
        row = np.concatenate([coeff_x, [coeff_M]])
        
        # RHS: - sum_i w_i * A[e,i]
        rhs = -np.sum(weights * A[e, :])
        
        # Constraint: row @ vars <= rhs
        # milp uses LinearConstraint(A_ub, lb, ub)
        # Here we have one row.
        A_ub_row = row.reshape(1, -1)
        lb_c = -np.inf
        ub_c = rhs
        constraints.append(LinearConstraint(A_ub_row, lb_c, ub_c))
        
    # Solve
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.success:
        x = res.x[:D]
        # Return 0 for CW (x < 0.5) and 1 for CCW (x >= 0.5)
        # The problem asks for an array where >= 0.5 is CCW.
        # Our x is 0 for CW, 1 for CCW.
        return x.astype(float)
    else:
        # Fallback to all clockwise if solver fails (should not happen for this small problem)
        return np.zeros(D)
