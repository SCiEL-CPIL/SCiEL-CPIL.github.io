# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        value >= 0.5 -> route the demand counter-clockwise
        value < 0.5  -> route the demand clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 630s limit, keep 30s margin

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Build edge-load matrices.
    # For each demand d and edge e:
    #   CW[d, e] = weight if edge e is on clockwise path from s to t, else 0
    #   CCW[d, e] = weight if edge e is on counter-clockwise path, else 0
    CW = np.zeros((D, n))
    CCW = np.zeros((D, n))

    for d, (s, t, w) in enumerate(demands):
        # Clockwise path from s to t: edges s, s+1, ..., t-1 (mod n)
        # If s == t, no edges (but in a ring, s!=t always here)
        # Number of clockwise edges:
        cw_len = (t - s) % n
        if cw_len == 0:
            cw_len = n  # full circle, but shouldn't happen with distinct nodes
        for i in range(cw_len):
            e = (s + i) % n
            CW[d, e] = w
            CCW[d, e] = 0
        # Counter-clockwise: all other edges
        for i in range(n):
            if CW[d, i] == 0:
                CCW[d, i] = w

    # ILP formulation:
    # Variables: x[0..D-1] in {0,1} (1 = CCW), t >= 0 (max load)
    # Minimize: t
    # Subject to: for each edge e: sum_d (CW[d,e]*(1-x[d]) + CCW[d,e]*x[d]) <= t
    #            <=> sum_d CW[d,e] - sum_d CW[d,e]*x[d] + sum_d CCW[d,e]*x[d] <= t
    #            <=> sum_d (CCW[d,e] - CW[d,e]) * x[d] - t <= -sum_d CW[d,e]
    #            <=> sum_d (CW[d,e] - CCW[d,e]) * x[d] + t >= sum_d CW[d,e]
    #
    # Let A[d,e] = CW[d,e] - CCW[d,e]
    # Constraint for edge e: sum_d A[d,e]*x[d] + t >= CW_sum[e]
    # where CW_sum[e] = sum_d CW[d,e]
    #
    # In standard form for scipy.optimize.milp (minimize c^T v subject to constraints):
    # Variables v = [x_0, ..., x_{D-1}, t]
    # Objective: c = [0, ..., 0, 1]
    # Constraints: for each edge e: -sum_d A[d,e]*x[d] - t <= -CW_sum[e]
    #              i.e., -A[e,:].@x - t <= -CW_sum[e]
    #
    # Also: 0 <= x[d] <= 1, x[d] integer; t >= 0

    A = CW - CCW  # shape (D, n), A[d,e] = CW[d,e] - CCW[d,e]
    CW_sum = CW.sum(axis=0)  # shape (n,)

    # Number of variables: D + 1
    nv = D + 1

    # Objective: minimize t
    c = np.zeros(nv)
    c[D] = 1.0  # coefficient for t

    # Constraints: for each edge e (n constraints):
    # -A[:, e].@x - t <= -CW_sum[e]
    # Coefficient matrix shape: (n, nv)
    constraint_matrix = np.zeros((n, nv))
    for e in range(n):
        constraint_matrix[e, :D] = -A[:, e]
        constraint_matrix[e, D] = -1.0  # coefficient for t

    constraint_lb = np.full(n, -np.inf)
    constraint_ub = -CW_sum  # right-hand side

    # Variable bounds:
    # x[d]: 0 <= x[d] <= 1, integer
    # t: 0 <= t < inf
    bounds = Bounds(
        lb=np.concatenate([np.zeros(D), [0.0]]),
        ub=np.concatenate([np.ones(D), [np.inf]])
    )

    # Integrality: x[0..D-1] are binary (integer), t is continuous
    integrality = np.concatenate([np.ones(D, dtype=int), [0]])

    # Solve ILP
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(constraint_matrix, constraint_lb, constraint_ub),
            integrality=integrality,
            bounds=bounds
        )
        if res.status == 0:
            x_opt = res.x[:D]
            routing = np.where(x_opt >= 0.5, 1.0, 0.0)
            return routing
    except Exception:
        pass

    # Fallback: try all-clockwise if ILP fails or times out
    if time.time() > deadline - 10:
        return np.zeros(D)

    # Simple heuristic fallback: greedy
    routing = np.zeros(D)
    for d in range(D):
        # Try both directions, pick the one that minimizes max load
        best_dir = 0
        best_max_load = np.inf
        for direction in [0, 1]:
            routing[d] = direction
            # Compute loads
            loads = np.zeros(n)
            for d2 in range(D):
                if routing[d2] < 0.5:
                    loads += CW[d2, :]
                else:
                    loads += CCW[d2, :]
            ml = loads.max()
            if ml < best_max_load:
                best_max_load = ml
                best_dir = direction
        routing[d] = best_dir

    return routing


# EVOLVE-BLOCK-END
