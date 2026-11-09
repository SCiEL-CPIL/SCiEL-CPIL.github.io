# EVOLVE-BLOCK-START
import numpy as np
import time

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
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute CW edge masks as bitmask
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod 10)
    cw_masks = []
    for s, t, w in demands:
        mask = 0
        if s <= t:
            for e in range(s, t):
                mask |= (1 << e)
        else:
            for e in range(s, N):
                mask |= (1 << e)
            for e in range(0, t):
                mask |= (1 << e)
        cw_masks.append(mask)
    
    # CCW mask is the complement within N bits
    all_edges_mask = (1 << N) - 1
    ccw_masks = [all_edges_mask ^ m for m in cw_masks]
    
    weights = np.array([w for s, t, w in demands], dtype=np.float64)
    
    def compute_max_load(routing):
        """Compute max edge load given routing decisions (0=CW, 1=CCW)."""
        loads = np.zeros(N, dtype=np.float64)
        for i in range(D):
            if routing[i] < 0.5:
                mask = cw_masks[i]
            else:
                mask = ccw_masks[i]
            for e in range(N):
                if mask & (1 << e):
                    loads[e] += weights[i]
        return np.max(loads)
    
    # Start with all clockwise
    best_routing = np.zeros(D)
    best_load = compute_max_load(best_routing)
    
    start_time = time.time()
    deadline = start_time + 620.0  # 10 second safety margin
    
    # Local search: try flipping one demand at a time
    improved = True
    while improved:
        improved = False
        for i in range(D):
            if time.time() > deadline:
                break
            # Try flipping demand i
            trial = best_routing.copy()
            trial[i] = 1.0 - trial[i]
            load = compute_max_load(trial)
            if load < best_load - 1e-9:
                best_routing = trial
                best_load = load
                improved = True
        if time.time() > deadline:
            break
    
    # If we still have time, do more extensive search
    if time.time() < deadline - 5.0:
        # Try random restarts with local search
        np.random.seed(42)
        for restart in range(50):
            if time.time() > deadline - 5.0:
                break
            # Random initial solution
            trial = np.random.choice([0.0, 1.0], size=D)
            load = compute_max_load(trial)
            
            # Local search from this starting point
            local_improved = True
            while local_improved:
                if time.time() > deadline - 5.0:
                    break
                local_improved = False
                for i in range(D):
                    if time.time() > deadline - 5.0:
                        break
                    trial_copy = trial.copy()
                    trial_copy[i] = 1.0 - trial_copy[i]
                    trial_load = compute_max_load(trial_copy)
                    if trial_load < load - 1e-9:
                        trial = trial_copy
                        load = trial_load
                        local_improved = True
                if time.time() > deadline - 5.0:
                    break
            
            if load < best_load - 1e-9:
                best_routing = trial
                best_load = load
    
    return best_routing

# EVOLVE-BLOCK-END
