import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
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
    
    # Precompute edge usage matrices
    # c0[i, e] is 1 if edge e is used by demand i in CW direction
    # c1[i, e] is 1 if edge e is used by demand i in CCW direction
    c0 = np.zeros((D, N), dtype=float)
    c1 = np.zeros((D, N), dtype=float)
    weights = np.zeros(D, dtype=float)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        # CW
        curr = s
        while curr != t:
            c0[i, curr] = 1.0
            curr = (curr + 1) % N
        # CCW
        curr = s
        while curr != t:
            curr = (curr - 1) % N
            c1[i, curr] = 1.0

    # Variables: y_0 ... y_{D-1}, L_max
    # Total variables: D + 1
    n_vars = D + 1
    
    # Objective: Minimize L_max
    # c = [0, ..., 0, 1]
    c = np.zeros(n_vars)
    c[D] = 1.0
    
    # Constraints:
    # For each edge e:
    # sum_i w_i * (c1[i,e] - c0[i,e]) * y_i - L_max <= - sum_i w_i * c0[i,e]
    
    A = np.zeros((N, n_vars))
    b = np.zeros(N)
    
    for e in range(N):
        for i in range(D):
            A[e, i] = weights[i] * (c1[i, e] - c0[i, e])
        A[e, D] = -1.0
        b[e] = -np.sum(weights * c0[:, e])
    
    # Bounds
    # y_i in [0, 1]
    # L_max in [0, inf]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[D] = np.inf
    
    # Integrality
    integrality = np.zeros(n_vars)
    integrality[:D] = 1.0 # y_i are binary
    
    constraints = LinearConstraint(A, ub=b, lb=-np.inf)
    bounds = Bounds(lb, ub)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if not res.success:
        # Fallback to a simple heuristic if solver fails, though it shouldn't
        # Just return all CW (0)
        return np.zeros(D)
    
    y = res.x[:D]
    routing = np.where(y >= 0.5, 1.0, 0.0)
    
    return routing
