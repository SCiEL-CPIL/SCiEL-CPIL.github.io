# EVOLVE-BLOCK-START
import time
import numpy as np

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


def _compute_max_load(routing: np.ndarray) -> float:
    """
    Compute the maximum edge load given a routing decision.
    routing: array of 0s and 1s (0=clockwise, 1=counter-clockwise)
    """
    # Load from clockwise demands
    # routing == 0 means clockwise
    # We can compute load as: for each edge, sum of weights of demands routed clockwise through it
    # + sum of weights of demands routed counter-clockwise through it
    
    # Clockwise contribution: CW_ROUTES * (routing == 0) * weights
    # Counter-clockwise contribution: (1 - CW_ROUTES) * (routing == 1) * weights
    
    cw_mask = (routing < 0.5)  # clockwise
    ccw_mask = ~cw_mask  # counter-clockwise
    
    # Load per edge
    # cw_load[e] = sum over demands i where cw_mask[i] and CW_ROUTES[i,e]==1 of WEIGHTS[i]
    # ccw_load[e] = sum over demands i where ccw_mask[i] and CW_ROUTES[i,e]==0 of WEIGHTS[i]
    
    # Vectorized: 
    # cw_contrib = CW_ROUTES * (cw_mask * WEIGHTS)[:, None]
    # ccw_contrib = (1 - CW_ROUTES) * (ccw_mask * WEIGHTS)[:, None]
    # total_load = cw_contrib.sum(axis=0) + ccw_contrib.sum(axis=0)
    
    cw_weights = np.where(cw_mask, WEIGHTS, 0.0)
    ccw_weights = np.where(ccw_mask, WEIGHTS, 0.0)
    
    cw_load = CW_ROUTES.T @ cw_weights
    ccw_load = (1 - CW_ROUTES).T @ ccw_weights
    total_load = cw_load + ccw_load
    
    return float(np.max(total_load))


def _greedy_initial_solution() -> np.ndarray:
    """
    Greedy construction: start with all clockwise, then flip demands to reduce max load.
    """
    routing = np.zeros(D)
    current_max = _compute_max_load(routing)
    
    # Try flipping each demand and keep the best
    for _ in range(3):  # multiple passes
        improved = False
        for i in range(D):
            routing[i] = 1.0 - routing[i]
            new_max = _compute_max_load(routing)
            if new_max < current_max - 1e-9:
                current_max = new_max
                improved = True
            else:
                routing[i] = 1.0 - routing[i]  # revert
        if not improved:
            break
    
    return routing


def _local_search(routing: np.ndarray, time_limit: float) -> np.ndarray:
    """
    Local search with first-improvement and occasional random restarts.
    """
    start_time = time.time()
    current_routing = routing.copy()
    current_max = _compute_max_load(current_routing)
    best_routing = current_routing.copy()
    best_max = current_max
    
    # Simulated annealing-like approach
    rng = np.random.default_rng(42)
    temp = 2.0
    cooling = 0.995
    
    step = 0
    while time.time() - start_time < time_limit:
        step += 1
        # Pick a random demand to flip
        i = rng.integers(0, D)
        current_routing[i] = 1.0 - current_routing[i]
        new_max = _compute_max_load(current_routing)
        
        delta = new_max - current_max
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            current_max = new_max
            if current_max < best_max - 1e-12:
                best_max = current_max
                best_routing = current_routing.copy()
        else:
            current_routing[i] = 1.0 - current_routing[i]  # revert
        
        temp *= cooling
        
        # Every 5000 steps, try a 2-opt style move
        if step % 5000 == 0:
            i = rng.integers(0, D)
            j = rng.integers(0, D)
            if i != j:
                current_routing[i] = 1.0 - current_routing[i]
                current_routing[j] = 1.0 - current_routing[j]
                new_max = _compute_max_load(current_routing)
                delta = new_max - current_max
                if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
                    current_max = new_max
                    if current_max < best_max - 1e-12:
                        best_max = current_max
                        best_routing = current_routing.copy()
                else:
                    current_routing[i] = 1.0 - current_routing[i]
                    current_routing[j] = 1.0 - current_routing[j]
    
    return best_routing


def _exhaustive_search() -> np.ndarray:
    """
    Exhaustive search over all 2^15 = 32768 possibilities.
    """
    best_routing = np.zeros(D)
    best_max = float('inf')
    
    for mask in range(1 << D):
        routing = np.zeros(D)
        for i in range(D):
            routing[i] = (mask >> i) & 1
        m = _compute_max_load(routing)
        if m < best_max:
            best_max = m
            best_routing = routing.copy()
    
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
    deadline = start_time + 600.0  # 600 second budget
    
    # Phase 1: Greedy initialization
    routing = _greedy_initial_solution()
    current_max = _compute_max_load(routing)
    
    # Phase 2: Local search / simulated annealing with remaining time
    remaining = deadline - time.time() - 5.0  # 5s safety margin
    if remaining > 1.0:
        routing = _local_search(routing, remaining)
    
    # Phase 3: If we have plenty of time left, do exhaustive search
    remaining = deadline - time.time() - 2.0
    if remaining > 10.0:
        # Check if exhaustive is feasible (it is: 32768 * ~microseconds)
        exhaustive = _exhaustive_search()
        ex_max = _compute_max_load(exhaustive)
        if ex_max < current_max - 1e-12:
            routing = exhaustive
            current_max = ex_max
    
    # Final safety: ensure valid output
    routing = np.where(routing >= 0.5, 1.0, 0.0).astype(np.float64)
    
    return routing


# EVOLVE-BLOCK-END
