# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def _compute_max_load(routing, demands, cw, weights):
    """Compute the maximum edge load for a given routing."""
    n = len(cw[0])
    D = len(demands)
    ccw = 1.0 - cw
    delta = ccw - cw
    # load[e] = base_load[e] + sum_d w_d * delta[d,e] * x_d
    base_load = cw.T @ weights
    x = routing.astype(np.float64)
    load = base_load + (delta * weights[:, np.newaxis]).T @ x
    return np.max(load)

def _greedy_local_search(demands, cw, weights, max_iters=2000):
    """Greedy + local search fallback."""
    n = len(cw[0])
    D = len(demands)
    ccw = 1.0 - cw
    delta = ccw - cw
    base_load = cw.T @ weights
    
    # Start with all clockwise
    x = np.zeros(D)
    best_load = _compute_max_load(x, demands, cw, weights)
    best_x = x.copy()
    
    # Greedy: for each demand, try flipping if it reduces max load
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for d in range(D):
            x[d] = 1.0 - x[d]
            new_load = _compute_max_load(x, demands, cw, weights)
            if new_load < best_load - 1e-9:
                best_load = new_load
                best_x = x.copy()
                improved = True
            else:
                x[d] = 1.0 - x[d]  # revert
    
    # Multiple restarts with random initializations
    rng = np.random.default_rng(42)
    for restart in range(50):
        x = rng.random(D) > 0.5
        cur_load = _compute_max_load(x, demands, cw, weights)
        if cur_load < best_load:
            best_load = cur_load
            best_x = x.copy()
        # Local search from this start
        for _ in range(100):
            best_improve = False
            for d in range(D):
                x[d] = 1.0 - x[d]
                new_load = _compute_max_load(x, demands, cw, weights)
                if new_load < best_load - 1e-9:
                    best_load = new_load
                    best_x = x.copy()
                    best_improve = True
                else:
                    x[d] = 1.0 - x[d]
            if not best_improve:
                break
    
    return best_x.astype(np.float64)

def ring_routing() -> np.ndarray:
    n = 10
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
    D = len(demands)
    weights = np.array([d[2] for d in demands], dtype=np.float64)

    # Precompute clockwise edge-usage matrix
    cw = np.zeros((D, n), dtype=np.float64)
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            for e in range(s, t):
                cw[i, e] = 1.0
        else:
            for e in range(s, n):
                cw[i, e] = 1.0
            for e in range(0, t):
                cw[i, e] = 1.0

    ccw = 1.0 - cw
    delta = ccw - cw
    w_delta = delta * weights[:, np.newaxis]  # (D, n)

    base_load = cw.T @ weights  # (n,)

    # ILP formulation:
    # Variables: x[0..D-1] binary (1 = CCW, 0 = CW), t continuous >= 0
    # Minimize t
    # Subject to: for each edge e: base_load[e] + sum_d w_delta[d,e] * x_d <= t
    #            => sum_d w_delta[d,e] * x_d - t <= -base_load[e]
    
    c = np.zeros(D + 1)
    c[D] = 1.0

    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    for e in range(n):
        A_ub[e, :D] = w_delta[:, e]
        A_ub[e, D] = -1.0
        b_ub[e] = -base_load[e]

    lb = np.zeros(D + 1)
    ub = np.full(D + 1, np.inf)
    ub[:D] = 1.0
    bounds = Bounds(lb, ub)

    integrality = np.zeros(D + 1, dtype=np.int8)
    integrality[:D] = 1

    constraint = LinearConstraint(A_ub, -np.inf, b_ub)

    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill

    try:
        result = milp(
            c=c,
            constraints=[constraint],
            integrality=integrality,
            bounds=bounds,
        )
    except Exception:
        result = None

    elapsed = time.time() - start_time

    if result is not None and result.success and result.x is not None:
        x = result.x[:D]
        routing = np.round(x).astype(np.float64)
        # Verify the solution is valid
        if time.time() < deadline - 10:
            # Polish: verify and potentially improve with local search
            cur_load = _compute_max_load(routing, demands, cw, weights)
            improved_x = _greedy_local_search(demands, cw, weights, max_iters=500)
            imp_load = _compute_max_load(improved_x, demands, cw, weights)
            if imp_load < cur_load - 1e-9:
                return improved_x
        return routing
    else:
        # Fallback: greedy local search
        if time.time() < deadline - 10:
            return _greedy_local_search(demands, cw, weights, max_iters=2000)
        return np.zeros(D)

# EVOLVE-BLOCK-END
