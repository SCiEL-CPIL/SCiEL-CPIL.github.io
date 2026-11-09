# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        value < 0.5 -> route the demand clockwise
        value >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one value per demand.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin for 630s limit

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute the edge incidence for each demand in each direction
    # clockwise_edges[d] = set of edges used if demand d goes clockwise
    # ccw_edges[d] = set of edges used if demand d goes counter-clockwise
    # Edge e connects node e to (e+1) mod n
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
    # Counter-clockwise from s to t: all other edges

    cw_edges = []
    ccw_edges = []
    for (s, t, w) in demands:
        cw_set = set()
        e = s
        while e != t:
            cw_set.add(e)
            e = (e + 1) % n
        cw_edges.append(cw_set)
        
        # Counter-clockwise uses all other edges
        all_edges = set(range(n))
        ccw_set = all_edges - cw_set
        ccw_edges.append(ccw_set)

    # Build the LP:
    # Variables: x_d for each demand d (binary: 0 = clockwise, 1 = counter-clockwise), and L (max load)
    # Minimize L
    # Subject to: for each edge e, sum over demands d of w_d * (x_d if e in ccw_edges[d] else (1-x_d) if e in cw_edges[d] else 0) <= L
    #             x_d in {0, 1}
    
    # Actually, let's formulate more carefully.
    # If x_d = 0 (clockwise), demand d uses edges in cw_edges[d].
    # If x_d = 1 (counter-clockwise), demand d uses edges in ccw_edges[d].
    
    # Load on edge e = sum_d w_d * (1 - x_d) * [e in cw_edges[d]] + sum_d w_d * x_d * [e in ccw_edges[d]]
    # We want: Load_e <= L for all e
    
    # Let's define the coefficient matrix A_ub for the constraint: Load_e - L <= 0
    # Load_e = sum_d w_d * (1-x_d) * I(e in cw_d) + sum_d w_d * x_d * I(e in ccw_d)
    #        = sum_d w_d * I(e in cw_d) - sum_d w_d * x_d * I(e in cw_d) + sum_d w_d * x_d * I(e in ccw_d)
    #        = C_e + sum_d x_d * w_d * (I(e in ccw_d) - I(e in cw_d))
    # where C_e = sum_d w_d * I(e in cw_d)
    
    # So constraint: C_e + sum_d x_d * w_d * (I(e in ccw_d) - I(e in cw_d)) - L <= 0
    # Variables: [x_0, x_1, ..., x_{D-1}, L]
    # Total variables: D + 1
    
    # Objective: minimize L -> c = [0, 0, ..., 0, 1]
    
    # A_ub has n rows (one per edge)
    # A_ub[e, d] = w_d * (I(e in ccw_edges[d]) - I(e in cw_edges[d])) for d in 0..D-1
    # A_ub[e, D] = -1 (coefficient of L)
    # b_ub = -C_e for each edge e
    
    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    
    for e in range(n):
        C_e = 0.0
        for d in range(D):
            w = demands[d][2]
            in_cw = 1.0 if e in cw_edges[d] else 0.0
            in_ccw = 1.0 if e in ccw_edges[d] else 0.0
            A_ub[e, d] = w * (in_ccw - in_cw)
            C_e += w * in_cw
        A_ub[e, D] = -1.0
        b_ub[e] = -C_e
    
    # Objective: minimize L
    c = np.zeros(D + 1)
    c[D] = 1.0
    
    # Bounds: 0 <= x_d <= 1, L >= 0
    bounds = [(0, 1)] * D + [(0, None)]
    
    # Solve using HiGHS
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='HiGHS')
    
    if result.status == 0:
        # Extract x values
        x = result.x[:D]
        # Round to nearest integer (should be close to 0 or 1)
        routing = np.round(x)
        # Ensure values are 0 or 1
        routing[routing < 0.5] = 0.0
        routing[routing >= 0.5] = 1.0
        return routing
    else:
        # Fallback: return all clockwise (zeros)
        return np.zeros(D)

# EVOLVE-BLOCK-END
