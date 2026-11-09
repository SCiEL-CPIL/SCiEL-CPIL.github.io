# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

# Fixed problem data
N = 10
DEMANDS = [
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
D = len(DEMANDS)

# Precompute route incidence vectors for clockwise routing
# For demand (s, t, w), clockwise uses edges s, s+1, ..., t-1 (mod N)
CW_ROUTES = np.zeros((D, N), dtype=np.int8)
for i, (s, t, w) in enumerate(DEMANDS):
    dist = (t - s) % N
    for k in range(dist):
        CW_ROUTES[i, (s + k) % N] = 1

# Precompute weights vector
WEIGHTS = np.array([w for _, _, w in DEMANDS], dtype=np.float64)

# Precompute the delta matrix: Delta[i, e] = w_i if edge e is NOT in CW path, else -w_i
# This is used in the LP formulation where x_i=1 means CCW routing
# Load[e] = sum_i w_i * (1 if CW else 0) + sum_i w_i * (1 if CCW else 0)
# If x_i = 1 (CCW): contribution is w_i for non-CW edges
# If x_i = 0 (CW): contribution is w_i for CW edges
# Total load[e] = sum_i w_i * (CW_ROUTES[i,e] * (1-x_i) + (1-CW_ROUTES[i,e]) * x_i)
#               = sum_i w_i * CW_ROUTES[i,e] + sum_i w_i * x_i * (1 - 2*CW_ROUTES[i,e])
# So: Load = Base + Delta @ x
# where Base[e] = sum_i w_i * CW_ROUTES[i,e]
#       Delta[e, i] = w_i * (1 - 2*CW_ROUTES[i,e])

Base_load = CW_ROUTES.T @ WEIGHTS  # shape (N,)
Delta = np.zeros((N, D), dtype=np.float64)
for e in range(N):
    for i in range(D):
        Delta[e, i] = WEIGHTS[i] * (1 - 2 * CW_ROUTES[i, e])


def _compute_max_load(routing: np.ndarray) -> float:
    """
    Compute the maximum edge load given a routing decision.
    routing: array of 0s and 1s (0=clockwise, 1=counter-clockwise)
    """
    cw_mask = (routing < 0.5)
    ccw_mask = ~cw_mask
    
    cw_weights = np.where(cw_mask, WEIGHTS, 0.0)
    ccw_weights = np.where(ccw_mask, WEIGHTS, 0.0)
    
    cw_load = CW_ROUTES.T @ cw_weights
    ccw_load = (1 - CW_ROUTES).T @ ccw_weights
    total_load = cw_load + ccw_load
    
    return float(np.max(total_load))


def _lp_solve() -> np.ndarray:
    """
    Solve the MILP formulation using linprog.
    Variables: x_0, ..., x_{D-1} (binary), t (continuous)
    Minimize: t
    Subject to:
      Load[e] <= t for all e
      Load[e] = Base[e] + sum_i Delta[e,i] * x_i
      x_i in {0, 1}
    
    Using linprog with bounds [0,1] for x and integrality constraints.
    """
    # Variables: [x_0, ..., x_{D-1}, t]
    n_vars = D + 1
    t_idx = D  # index of t in the variable vector
    
    # Objective: minimize t
    c = np.zeros(n_vars)
    c[t_idx] = 1.0
    
    # Constraints:
    # For each edge e: Load[e] <= t
    # Base[e] + sum_i Delta[e,i] * x_i <= t
    # Base[e] + sum_i Delta[e,i] * x_i - t <= 0
    A_ub = np.zeros((N, n_vars))
    b_ub = np.zeros(N)
    
    for e in range(N):
        A_ub[e, :D] = Delta[e, :]
        A_ub[e, t_idx] = -1.0
        b_ub[e] = -Base_load[e]
    
    # Bounds: x_i in [0, 1], t in [0, inf)
    bounds = [(0, 1)] * D + [(0, None)]
    
    # Integrality: x_0, ..., x_{D-1} are integers
    integrality = [1] * D + [0]
    
    # Solve using linprog with method='highs' (supports integer variables)
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        integrality=integrality,
        method='highs'
    )
    
    if result.status == 0:
        x = result.x[:D]
        # Round to nearest 0 or 1
        routing = np.where(x >= 0.5, 1.0, 0.0)
        return routing
    else:
        # Fallback to exhaustive search if LP fails
        return _exhaustive_search()


def _exhaustive_search() -> np.ndarray:
    """
    Exhaustive search over all 2^15 = 32768 possibilities using vectorized numpy.
    """
    masks = np.arange(1 << D, dtype=np.int32)
    routing_matrix = np.zeros((1 << D, D), dtype=np.float64)
    for j in range(D):
        routing_matrix[:, j] = (masks >> j) & 1
    
    cw_mask = routing_matrix < 0.5
    ccw_mask = ~cw_mask
    
    cw_weights = cw_mask * WEIGHTS[np.newaxis, :]
    ccw_weights = ccw_mask * WEIGHTS[np.newaxis, :]
    
    cw_loads = cw_weights @ CW_ROUTES
    ccw_loads = ccw_weights @ (1 - CW_ROUTES)
    
    total_loads = cw_loads + ccw_loads
    max_loads = total_loads.max(axis=1)
    
    best_idx = np.argmin(max_loads)
    best_routing = routing_matrix[best_idx].copy()
    
    return best_routing


def _greedy_polish(routing: np.ndarray, time_limit: float) -> np.ndarray:
    """
    Greedy local search polish: try flipping each demand to reduce max load.
    """
    start_time = time.time()
    current_routing = routing.copy()
    current_max = _compute_max_load(current_routing)
    best_routing = current_routing.copy()
    best_max = current_max
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(D):
            current_routing[i] = 1.0 - current_routing[i]
            new_max = _compute_max_load(current_routing)
            if new_max < current_max - 1e-9:
                current_max = new_max
                if current_max < best_max - 1e-12:
                    best_max = current_max
                    best_routing = current_routing.copy()
                improved = True
            else:
                current_routing[i] = 1.0 - current_routing[i]
    
    return best_routing


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    
    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin
    
    # First, try the LP/MILP approach which should find the global optimum
    routing = _lp_solve()
    
    # Verify the solution and polish with greedy local search
    routing = _greedy_polish(routing, time_limit=30.0)
    
    # Final safety: ensure valid output
    routing = np.where(routing >= 0.5, 1.0, 0.0).astype(np.float64)
    
    return routing


# EVOLVE-BLOCK-END
