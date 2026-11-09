# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def _compute_edge_contribs(n=10, demands=None):
    """
    For each demand, compute which edges are used if routed clockwise (CW)
    vs counter-clockwise (CCW).
    Returns (cw, ccw) arrays of shape (D, n).
    """
    if demands is None:
        demands = [
            (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1),
            (0, 4, 5), (2, 8, 2), (5, 9, 3), (1, 4, 2),
            (6, 0, 4), (3, 8, 1), (7, 2, 3), (4, 9, 2),
            (8, 1, 5), (0, 6, 1), (5, 2, 2),
        ]
    D = len(demands)
    cw = np.zeros((D, n))
    ccw = np.zeros((D, n))
    for i, (s, t, w) in enumerate(demands):
        # CW: edges s, s+1, ..., t-1 (mod n)
        if s <= t:
            edges = range(s, t)
        else:
            edges = list(range(s, n)) + list(range(0, t))
        for e in edges:
            cw[i, e] = w
        # CCW: all other edges
        for e in range(n):
            if cw[i, e] == 0:
                ccw[i, e] = w
    return cw, ccw


def _max_load(routing, cw, ccw):
    """
    routing: array of 0/1, 0=CW, 1=CCW
    Returns the maximum edge load.
    """
    loads = np.zeros(cw.shape[1])
    for i in range(len(routing)):
        if routing[i] >= 0.5:
            loads += ccw[i]
        else:
            loads += cw[i]
    return np.max(loads)


def ring_routing() -> np.ndarray:
    t_start = time.time()
    deadline = t_start + 590.0  # safety margin

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1),
        (0, 4, 5), (2, 8, 2), (5, 9, 3), (1, 4, 2),
        (6, 0, 4), (3, 8, 1), (7, 2, 3), (4, 9, 2),
        (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    cw, ccw = _compute_edge_contribs(n, demands)

    # Initial best: all CW
    best_routing = np.zeros(D)
    best_load = _max_load(best_routing, cw, ccw)

    # Also try all CCW
    alt = np.ones(D)
    alt_load = _max_load(alt, cw, ccw)
    if alt_load < best_load:
        best_routing = alt
        best_load = alt_load

    # MILP formulation:
    # Variables: x_0..x_{D-1} (binary, 1=CCW, 0=CW), L (continuous, max load)
    # Total variables: D + 1
    # Objective: minimize L
    # Constraints: for each edge e, sum_i (w_i * [if x_i=0 then cw[i,e] else ccw[i,e]]) <= L
    #   This is: for each edge e, sum_i (cw[i,e] * (1-x_i) + ccw[i,e] * x_i) <= L
    #   = sum_i (cw[i,e] - cw[i,e]*x_i + ccw[i,e]*x_i) <= L
    #   = sum_i cw[i,e] + sum_i (ccw[i,e] - cw[i,e]) * x_i <= L
    #   = sum_i cw[i,e] + sum_i delta[i,e] * x_i <= L
    #   where delta[i,e] = ccw[i,e] - cw[i,e]
    #   Rearranged: sum_i delta[i,e] * x_i - L <= -sum_i cw[i,e]

    # Build constraint matrix A for: A @ [x; L] <= b
    # x is D binary vars, L is 1 continuous var
    # Total vars = D + 1
    nvars = D + 1

    # Objective: minimize L -> c = [0, 0, ..., 0, 1]
    c = np.zeros(nvars)
    c[-1] = 1.0  # coefficient for L

    # Constraints: for each edge e (n constraints):
    # sum_i delta[i,e] * x_i - L <= -sum_i cw[i,e]
    A = np.zeros((n, nvars))
    b = np.zeros(n)
    for e in range(n):
        for i in range(D):
            delta = ccw[i, e] - cw[i, e]
            A[e, i] = delta
        A[e, -1] = -1.0  # coefficient for L
        b[e] = -np.sum(cw[:, e])

    # Variable bounds: x_i in [0,1] integer, L in [0, inf)
    integrality = np.zeros(nvars)
    integrality[:D] = 1  # x_i are binary (integer)
    integrality[-1] = 0  # L is continuous

    # Bounds for each variable
    var_bounds = Bounds(
        lb=np.concatenate([np.zeros(D), [0.0]]),
        ub=np.concatenate([np.ones(D), [np.inf]]),
    )

    # Linear constraint: A @ vars <= b
    constraint = LinearConstraint(A, ub=b)

    try:
        result = milp(
            c=c,
            constraints=[constraint],
            integrality=integrality,
            bounds=var_bounds,
        )
        if result.status == 0:  # optimal
            x_opt = np.round(result.x[:D])
            load = _max_load(x_opt, cw, ccw)
            if load < best_load:
                best_routing = x_opt
                best_load = load
    except Exception:
        pass

    # Local search: try flipping each bit, and pairs
    improved = True
    while improved:
        improved = False
        # Single-bit flips
        for i in range(D):
            if time.time() > deadline:
                break
            candidate = best_routing.copy()
            candidate[i] = 1.0 - candidate[i]
            load = _max_load(candidate, cw, ccw)
            if load < best_load - 1e-12:
                best_routing = candidate
                best_load = load
                improved = True
        if time.time() > deadline:
            break
        # Pair flips
        for i in range(D):
            if time.time() > deadline:
                break
            for j in range(i + 1, D):
                if time.time() > deadline:
                    break
                candidate = best_routing.copy()
                candidate[i] = 1.0 - candidate[i]
                candidate[j] = 1.0 - candidate[j]
                load = _max_load(candidate, cw, ccw)
                if load < best_load - 1e-12:
                    best_routing = candidate
                    best_load = load
                    improved = True
            if time.time() > deadline:
                break

    return best_routing


# EVOLVE-BLOCK-END
