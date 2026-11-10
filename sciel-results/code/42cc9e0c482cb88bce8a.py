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
        0 (or < 0.5) -> route the demand clockwise
        1 (or >= 0.5) -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge usage for each demand in both directions
    # cw_edges[d] is a boolean array of length n indicating which edges are used if demand d goes clockwise
    # ccw_edges[d] is the complement
    cw_edges = np.zeros((D, n), dtype=int)
    ccw_edges = np.zeros((D, n), dtype=int)
    
    for d_idx, (s, t, w) in enumerate(demands):
        # Clockwise: s -> s+1 -> ... -> t
        # Edges are (i, i+1 mod n) for i in [s, s+1, ..., t-1] mod n
        if s <= t:
            for i in range(s, t):
                cw_edges[d_idx, i] = 1
        else:
            for i in range(s, n):
                cw_edges[d_idx, i] = 1
            for i in range(0, t):
                cw_edges[d_idx, i] = 1
        
        # Counter-clockwise: all other edges
        ccw_edges[d_idx] = 1 - cw_edges[d_idx]
    
    weights = np.array([w for _, _, w in demands])
    
    def evaluate(routing):
        """Compute max edge load for a given routing (0=cw, 1=ccw)"""
        # routing: array of 0/1
        # load[e] = sum over d where routing[d]==0 of weights[d]*cw_edges[d,e] + sum over d where routing[d]==1 of weights[d]*ccw_edges[d,e]
        # We can compute this efficiently:
        # load = cw_edges.T @ (weights * (1-routing)) + ccw_edges.T @ (weights * routing)
        # Or simpler:
        # For each edge e, load[e] = sum_d weights[d] * (cw_edges[d,e] * (1-routing[d]) + ccw_edges[d,e] * routing[d])
        
        # Vectorized:
        # Let x = routing (0 or 1)
        # contribution from d to edge e is w[d] * (cw[d,e] if x[d]==0 else ccw[d,e])
        # = w[d] * (cw[d,e] * (1-x[d]) + ccw[d,e] * x[d])
        
        # We can precompute:
        # load = sum_d w[d] * (cw[d,:] * (1-x[d]) + ccw[d,:] * x[d])
        # = sum_d w[d] * (cw[d,:] + (ccw[d,:] - cw[d,:]) * x[d])
        # = sum_d w[d] * cw[d,:] + sum_d w[d] * (ccw[d,:] - cw[d,:]) * x[d]
        
        base_load = cw_edges.T @ weights  # load if all go CW
        diff = (ccw_edges - cw_edges).T @ (weights * routing)  # adjustment
        load = base_load + diff
        return np.max(load)
    
    # Start with all clockwise
    best_routing = np.zeros(D, dtype=int)
    best_score = evaluate(best_routing)
    
    # Local search: hill climbing
    # Try flipping each demand one at a time, accept if it improves
    improved = True
    while improved:
        improved = False
        for d_idx in range(D):
            # Try flipping
            trial = best_routing.copy()
            trial[d_idx] = 1 - trial[d_idx]
            trial_score = evaluate(trial)
            if trial_score < best_score:
                best_routing = trial
                best_score = trial_score
                improved = True
    
    # Convert to float array for return
    return best_routing.astype(float)

# EVOLVE-BLOCK-END
