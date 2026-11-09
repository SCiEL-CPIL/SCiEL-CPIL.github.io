# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand
        (0 = clockwise, 1 = counter-clockwise)
    """
    deadline = time.time() + 590  # safety margin before 600s kill

    # Problem setup
    n = 10  # nodes 0..9, edges 0..9 (edge e connects node e to node (e+1)%10)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute edge masks for clockwise routing
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod 10)
    cw_mask = np.zeros((D, n), dtype=np.int32)
    ccw_mask = np.zeros((D, n), dtype=np.int32)
    weights = np.zeros(D, dtype=np.int32)

    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        # Clockwise edges: s, (s+1)%n, ..., (t-1)%n
        edge = s
        while True:
            cw_mask[i, edge] = 1
            if edge == (t - 1) % n:
                break
            edge = (edge + 1) % n
        # Counter-clockwise: all other edges
        ccw_mask[i] = 1 - cw_mask[i]

    def compute_loads(routing):
        """Compute edge loads given routing directions (0=cw, 1=ccw)."""
        loads = np.zeros(n, dtype=np.int32)
        for i in range(D):
            if routing[i] < 0.5:
                loads += cw_mask[i] * weights[i]
            else:
                loads += ccw_mask[i] * weights[i]
        return loads

    def max_load(routing):
        """Return the maximum edge load."""
        loads = compute_loads(routing)
        return int(np.max(loads))

    # Start with all clockwise
    best_routing = np.zeros(D)
    best_val = max_load(best_routing)

    # Also try all counter-clockwise
    alt = np.ones(D)
    alt_val = max_load(alt)
    if alt_val < best_val:
        best_routing = alt.copy()
        best_val = alt_val

    # Greedy local search: try flipping each demand, keep if it improves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(D):
            # Try flipping demand i
            trial = best_routing.copy()
            trial[i] = 1.0 - trial[i]
            val = max_load(trial)
            if val < best_val:
                best_routing = trial
                best_val = val
                improved = True
        # Second pass: try pairs of flips
        if not improved:
            for i in range(D):
                for j in range(i + 1, D):
                    trial = best_routing.copy()
                    trial[i] = 1.0 - trial[i]
                    trial[j] = 1.0 - trial[j]
                    val = max_load(trial)
                    if val < best_val:
                        best_routing = trial
                        best_val = val
                        improved = True

    # Simulated annealing / random restarts for further improvement
    best_overall = best_routing.copy()
    best_overall_val = best_val

    rng = np.random.RandomState(42)
    sa_temp = 5.0
    sa_cool = 0.995

    while time.time() < deadline:
        # Random restart
        current = rng.random(D)
        current[current < 0.5] = 0.0
        current[current >= 0.5] = 1.0
        curr_val = max_load(current)

        # Simulated annealing
        for step in range(2000):
            if time.time() > deadline:
                break
            i = rng.randint(D)
            trial = current.copy()
            trial[i] = 1.0 - trial[i]
            trial_val = max_load(trial)
            delta = trial_val - curr_val
            if delta <= 0 or rng.random() < np.exp(-delta / sa_temp):
                current = trial
                curr_val = trial_val
            sa_temp *= sa_cool
            if sa_temp < 0.01:
                break

        if curr_val < best_overall_val:
            best_overall = current.copy()
            best_overall_val = curr_val

    # Final greedy polish
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(D):
            trial = best_overall.copy()
            trial[i] = 1.0 - trial[i]
            val = max_load(trial)
            if val < best_overall_val:
                best_overall = trial
                best_overall_val = val
                improved = True

    return best_overall

# EVOLVE-BLOCK-END
