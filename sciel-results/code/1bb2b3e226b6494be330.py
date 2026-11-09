# EVOLVE-BLOCK-START
import numpy as np
import time
import random
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
    N = 10  # number of nodes/edges
    D = 15  # number of demands
    
    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute edge incidence for each demand
    # For each demand, precompute which edges are used if routed clockwise vs counter-clockwise
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod N)
    # Counter-clockwise uses all other edges
    
    cw_edges = []  # list of lists of edge indices for clockwise routing
    ccw_edges = []  # list of lists of edge indices for counter-clockwise routing
    
    for (s, t, w) in demands:
        # Clockwise edges
        cw = []
        if s <= t:
            cw = list(range(s, t))
        else:
            cw = list(range(s, N)) + list(range(0, t))
        cw_edges.append(cw)
        
        # Counter-clockwise edges (complement of clockwise)
        ccw = [e for e in range(N) if e not in cw]
        ccw_edges.append(ccw)
    
    # Convert to numpy arrays for faster operations
    cw_edge_arrs = [np.array(e) for e in cw_edges]
    ccw_edge_arrs = [np.array(e) for e in ccw_edges]
    weights = np.array([w for (_, _, w) in demands])
    
    def compute_max_load(routing):
        """Compute the maximum edge load given a routing decision vector."""
        loads = np.zeros(N)
        for i in range(D):
            if routing[i] >= 0.5:  # counter-clockwise
                if len(ccw_edge_arrs[i]) > 0:
                    loads[ccw_edge_arrs[i]] += weights[i]
            else:  # clockwise
                if len(cw_edge_arrs[i]) > 0:
                    loads[cw_edge_arrs[i]] += weights[i]
        return np.max(loads)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580  # 580 seconds, leaving 20s safety margin
    
    best_routing = np.zeros(D)
    best_score = compute_max_load(best_routing)
    
    # Strategy 1: Exhaustive search over all 2^15 = 32768 combinations
    # This is fast enough to do completely
    all_combinations = list(product([0, 1], repeat=D))
    
    for combo in all_combinations:
        routing = np.array(combo, dtype=float)
        score = compute_max_load(routing)
        if score < best_score:
            best_score = score
            best_routing = routing.copy()
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # If we finished exhaustive search, we have the optimal solution
    # Otherwise, do some local search to improve
    
    # Local search: try flipping individual bits and pairs of bits
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(D):
            trial = best_routing.copy()
            trial[i] = 1.0 - trial[i]
            score = compute_max_load(trial)
            if score < best_score - 1e-9:
                best_score = score
                best_routing = trial
                improved = True
        
        # Try flipping pairs of bits
        for i in range(D):
            for j in range(i + 1, D):
                trial = best_routing.copy()
                trial[i] = 1.0 - trial[i]
                trial[j] = 1.0 - trial[j]
                score = compute_max_load(trial)
                if score < best_score - 1e-9:
                    best_score = score
                    best_routing = trial
                    improved = True
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    return best_routing

# EVOLVE-BLOCK-END
