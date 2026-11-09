# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from itertools import product


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
        (5, 2, 2),
    ]

    # Build CW_use matrix: D x n
    CW_use = np.zeros((D, n), dtype=int)
    for d, (s, t, w) in enumerate(demands):
        if s == t:
            CW_use[d, :] = 1
        elif s < t:
            CW_use[d, s:t] = 1
        else:
            CW_use[d, s:] = 1
            CW_use[d, :t] = 1

    weights = np.array([w for _, _, w in demands], dtype=float)
    base_load = CW_use.T @ weights  # shape (n,)

    # Coefficient matrix A: n x D, where A[e, d] = w_d * (1 - 2 * CW_use[d][e])
    A = np.zeros((n, D), dtype=float)
    for d in range(D):
        A[:, d] = weights[d] * (1 - 2 * CW_use[d, :])

    # First solve the LP relaxation to get an initial solution and bound
    c = np.zeros(D + 1)
    c[0] = 1.0

    A_ub = np.zeros((n, D + 1))
    A_ub[:, 0] = -1.0
    A_ub[:, 1:] = A

    b_ub = -base_load

    bounds = [(0, None)] + [(0, 1)] * D

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not res.success:
        # Fallback: all clockwise
        return np.zeros(D)

    x_lp = res.x[1:]  # D values from LP solution
    L_lp = res.x[0]

    # Round the LP solution to get an initial integer solution
    routing_init = (x_lp >= 0.5).astype(float)

    # Compute the max load for the initial solution
    def compute_max_load(routing):
        loads = base_load + A @ routing
        return np.max(loads)

    best_routing = routing_init.copy()
    best_max_load = compute_max_load(best_routing)

    # Use branch and bound: try flipping individual bits to improve
    # This is a local search / hill climbing approach starting from the LP solution
    improved = True
    while improved:
        improved = False
        for d in range(D):
            # Try flipping bit d
            new_routing = best_routing.copy()
            new_routing[d] = 1.0 - new_routing[d]
            new_max_load = compute_max_load(new_routing)
            if new_max_load < best_max_load:
                best_routing = new_routing
                best_max_load = new_max_load
                improved = True

    # Try all 2^15 = 32768 combinations if it's fast enough
    # For 15 binary variables, this is feasible
    best_max_load_brute = best_max_load
    best_routing_brute = best_routing.copy()

    for combo in product([0, 1], repeat=D):
        routing = np.array(combo, dtype=float)
        max_load = compute_max_load(routing)
        if max_load < best_max_load_brute:
            best_max_load_brute = max_load
            best_routing_brute = routing

    # Use the better of the two
    if best_max_load_brute < best_max_load:
        best_routing = best_routing_brute

    return best_routing


# EVOLVE-BLOCK-END
