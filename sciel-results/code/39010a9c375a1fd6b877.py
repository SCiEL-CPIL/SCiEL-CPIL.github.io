# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Uses scipy.optimize.milp to find the optimal routing.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
        0 -> clockwise, 1 (or >= 0.5) -> counter-clockwise
    """
    n = 10
    D = 15
    
    # Demands: (source, target, weight)
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
    
    # Precompute cw_e_i and ccw_e_i
    cw = np.zeros((n, D), dtype=int)
    ccw = np.zeros((n, D), dtype=int)
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise edges: s, s+1, ..., t-1 (mod n)
        if s <= t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n)) + list(range(0, t))
        
        for e in cw_edges:
            cw[e, i] = 1
        
        # Counter-clockwise edges: all other edges
        for e in range(n):
            if cw[e, i] == 0:
                ccw[e, i] = 1
    
    # Now set up the MILP
    # Number of variables: D (binary) + 1 (continuous M)
    # Variable order: x_0, x_1, ..., x_{D-1}, M
    
    # Objective: minimize M
    c = np.zeros(D + 1)
    c[D] = 1.0  # M is the last variable
    
    # Constraints for each edge e:
    #   sum_i weight_i * (ccw_e_i - cw_e_i) * x_i - M <= -sum_i weight_i * cw_e_i
    
    A = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    
    weights = np.array([d[2] for d in demands])
    
    for e in range(n):
        for i in range(D):
            w = demands[i][2]
            A[e, i] = w * (ccw[e, i] - cw[e, i])
        A[e, D] = -1.0  # coefficient for M
        b_ub[e] = -np.sum(weights * cw[e, :])
    
    # Variable bounds: x_i in [0, 1], M >= 0
    lb = np.zeros(D + 1)
    ub = np.ones(D + 1)
    
    # Upper bound for M: sum of all weights is a safe upper bound
    total_weight = np.sum(weights)
    ub[D] = total_weight
    
    # Integrality: x_i are binary, M is continuous
    integrality = np.zeros(D + 1)
    integrality[:D] = 1  # binary
    integrality[D] = 0   # continuous
    
    # Linear constraint: A @ vars <= b_ub
    constraint = LinearConstraint(A, ub=b_ub)
    bounds = Bounds(lb, ub)
    
    # Solve
    # Enable presolve and set a reasonable time limit
    options = {
        'presolve': True,
        'time_limit': 50,
        'mip_rel_gap': 1e-6
    }
    
    res = milp(
        c=c,
        constraints=[constraint],
        integrality=integrality,
        bounds=bounds,
        options=options
    )
    
    if res.status == 0:  # optimal
        x = res.x[:D]
        # Round to be safe
        routing = np.round(x).astype(float)
        # Ensure values are 0 or 1
        routing[routing > 0.5] = 1.0
        routing[routing <= 0.5] = 0.0
        return routing
    else:
        # Fallback: return all clockwise (zeros)
        return np.zeros(D)


# EVOLVE-BLOCK-END
