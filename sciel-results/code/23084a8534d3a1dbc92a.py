# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

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
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute edge incidence for each demand in each direction
    # For demand (s, t, w):
    #   Clockwise (cw): uses edges s, s+1, ..., t-1 (mod 10)
    #   Counter-clockwise (ccw): uses all other edges

    # A[d, e] = 1 if demand d uses edge e when routed clockwise
    # B[d, e] = 1 if demand d uses edge e when routed counter-clockwise
    A = np.zeros((D, n))
    B = np.zeros((D, n))

    for d, (s, t, w) in enumerate(demands):
        # Clockwise path: edges s, s+1, ..., t-1 (mod 10)
        if s <= t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n)) + list(range(0, t))
        for e in cw_edges:
            A[d, e] = 1

        # Counter-clockwise path: all edges NOT in clockwise path
        for e in range(n):
            if A[d, e] == 0:
                B[d, e] = 1

    weights = np.array([w for (_, _, w) in demands], dtype=float)

    # LP formulation:
    # Variables: x_d in [0, 1] for d=0..D-1 (1 = ccw, 0 = cw), and L (max load)
    # Minimize L
    # Subject to: for each edge e:
    #   sum_d weights[d] * (A[d,e] * (1-x_d) + B[d,e] * x_d) <= L
    #   => sum_d weights[d] * A[d,e] + sum_d weights[d] * (B[d,e] - A[d,e]) * x_d <= L
    #   => sum_d weights[d] * (B[d,e] - A[d,e]) * x_d - L <= -sum_d weights[d] * A[d,e]

    # Decision vector: [x_0, x_1, ..., x_{D-1}, L]
    # Objective: minimize L -> c = [0, 0, ..., 0, 1]
    c = np.zeros(D + 1)
    c[D] = 1.0  # L is the last variable

    # Constraints: A_ub @ vars <= b_ub
    # For each edge e:
    #   sum_d weights[d] * (B[d,e] - A[d,e]) * x_d - L <= -sum_d weights[d] * A[d,e]
    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)

    for e in range(n):
        for d in range(D):
            A_ub[e, d] = weights[d] * (B[d, e] - A[d, e])
        A_ub[e, D] = -1.0
        b_ub[e] = -np.sum(weights * A[:, e])

    # Bounds: 0 <= x_d <= 1, L >= 0
    bounds = [(0, 1)] * D + [(0, None)]

    # Solve LP with HiGHS
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    # Check deadline
    if time.time() > deadline:
        # Return all-clockwise as fallback
        return np.zeros(D)

    if result.success:
        x = result.x[:D]
        # Round to binary
        routing = (x >= 0.5).astype(float)
    else:
        # Fallback: all clockwise
        routing = np.zeros(D)

    # Verify and polish: check if we can improve with local search
    if time.time() < deadline - 10:
        # Local search: try flipping each demand and see if max load decreases
        def compute_max_load(r):
            load = np.zeros(n)
            for d in range(D):
                if r[d] >= 0.5:
                    load += weights[d] * B[d]
                else:
                    load += weights[d] * A[d]
            return np.max(load)

        best_routing = routing.copy()
        best_load = compute_max_load(best_routing)

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for d in range(D):
                trial = best_routing.copy()
                trial[d] = 1.0 - trial[d]
                trial_load = compute_max_load(trial)
                if trial_load < best_load - 1e-9:
                    best_routing = trial
                    best_load = trial_load
                    improved = True

        routing = best_routing

    return routing


# EVOLVE-BLOCK-END
