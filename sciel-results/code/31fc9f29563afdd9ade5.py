import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    
    # Precompute for each demand:
    #   cw_mask[d]: set of edges used if routed clockwise
    #   ccw_mask[d]: set of edges used if routed counter-clockwise
    # Edge e joins node e and (e+1) mod n.
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod n).
    # Counter-clockwise: the complementary set of edges (all edges not on cw path).

    cw_masks = []
    ccw_masks = []
    for (s, t, w) in demands:
        cw = set()
        e = s
        while e != t:
            cw.add(e % n)
            e = (e + 1) % n
        all_edges = set(range(n))
        ccw = all_edges - cw
        cw_masks.append(cw)
        ccw_masks.append(ccw)

    weights = np.array([w for (s, t, w) in demands], dtype=float)

    # Precompute edge-incidence vectors for each demand and direction
    # cw_vec[d]: binary vector of length n, 1 where edge is on CW path
    # ccw_vec[d]: binary vector of length n, 1 where edge is on CCW path
    cw_vecs = np.zeros((D, n), dtype=np.int32)
    ccw_vecs = np.zeros((D, n), dtype=np.int32)
    for d in range(D):
        for e in cw_masks[d]:
            cw_vecs[d, e] = 1
        for e in ccw_masks[d]:
            ccw_vecs[d, e] = 1

    # Formulate ILP
    # Variables: x[0..D-1] are binary (direction), x[D] is continuous (max load t)
    # Objective: Minimize x[D]
    # Constraints: For each edge e, load_e <= t
    # load_e = sum_d w_d * (cw_vecs[d,e] if x_d==0 else ccw_vecs[d,e])
    # load_e = sum_d w_d * (cw_vecs[d,e] + x_d * (ccw_vecs[d,e] - cw_vecs[d,e]))
    # load_e = base_load_e + sum_d x_d * delta_e_d
    # base_load_e = sum_d w_d * cw_vecs[d,e]
    # delta_e_d = w_d * (ccw_vecs[d,e] - cw_vecs[d,e])
    
    # Constraint: base_load_e + sum_d delta_e_d * x_d - t <= 0
    # Matrix A for scipy.optimize.milp:
    # Rows correspond to edges (n rows).
    # Columns correspond to variables (D + 1 columns).
    
    A = np.zeros((n, D + 1))
    b_ub = np.zeros(n) # RHS for constraints: A @ x <= b_ub
    
    for e in range(n):
        # Coefficients for x_d
        for d in range(D):
            A[e, d] = weights[d] * (ccw_vecs[d, e] - cw_vecs[d, e])
        # Coefficient for t (variable index D)
        A[e, D] = -1.0
        # RHS
        b_ub[e] = -np.sum(weights * cw_vecs[:, e]) # -base_load_e

    # Objective: c @ x, minimize
    c = np.zeros(D + 1)
    c[D] = 1.0 # Minimize t

    # Bounds
    # x_d in [0, 1] for d in 0..D-1
    # t in [0, inf)
    lb = np.zeros(D + 1)
    ub = np.ones(D + 1)
    ub[D] = np.inf
    
    # Integrality: x_d are integers (binary), t is continuous
    integrality = np.zeros(D + 1)
    integrality[:D] = 1
    integrality[D] = 0

    # Solve
    res = milp(c=c,
               constraints=LinearConstraint(A, -np.inf, b_ub),
               integrality=integrality,
               bounds=Bounds(lb, ub))
    
    if res.success:
        # Extract routing decisions (first D variables)
        routing = res.x[:D].astype(float)
        # Ensure values are strictly 0 or 1
        routing[routing > 0.5] = 1.0
        routing[routing <= 0.5] = 0.0
        return routing
    else:
        # Fallback to all clockwise if solver fails (should not happen)
        return np.zeros(D, dtype=float)

# EVOLVE-BLOCK-END
