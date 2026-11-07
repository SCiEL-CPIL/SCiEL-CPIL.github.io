"""Guided local search penalty rule for TSP with adaptive kick magnitude."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for safety deadline
_START_TIME = time.monotonic()
_DEADLINE_SECONDS = 290.0  # Safety margin before 330s kill limit


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Returns a modified distance matrix for the evaluator's 2-opt.
    
    Strategy:
    1. Estimate the current round number from the max value in edge_n_used.
    2. Calculate an adaptive penalty strength that starts high (to escape local optima)
       and decays as rounds progress (to allow refinement).
    3. Apply a smooth penalty to frequently-used edges, with special emphasis on
       edges in the current local-optimum tour.
    """
    # Check deadline - if we're too close to the limit, return a safe baseline
    elapsed = time.monotonic() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # Estimate round number from edge_n_used
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_used > 0:
        # Round estimate: the max count indicates how many times the most-used edge has appeared
        round_est = int(max_used)
    else:
        round_est = 0
    
    # Adaptive kick magnitude: starts at 0.3, decays to 0.05 over 20 rounds
    # Use a smooth decay: kick = base * exp(-decay_rate * round)
    base_kick = 0.25
    decay_rate = 0.12
    kick = base_kick * np.exp(-decay_rate * round_est)
    
    # Create a copy of the distance matrix
    modified = edge_distance.copy()
    
    # Build a weight matrix for penalties
    # Penalty is proportional to usage count, normalized
    if max_used > 0:
        # Normalize edge_n_used to [0, 1] range
        norm_used = edge_n_used / (1.0 + max_used)
    else:
        norm_used = np.zeros_like(edge_n_used)
    
    # Apply penalty: increase distances for frequently-used edges
    # Use a smooth, nonlinear penalty to avoid discontinuities
    # penalty_factor = 1 + kick * norm_used^alpha where alpha > 1 emphasizes heavy use
    alpha = 1.5
    penalty_factor = 1.0 + kick * (norm_used ** alpha)
    
    # Additionally, boost penalty for edges in the current local-optimum tour
    # These are the edges we most want to break
    tour_penalty_boost = 1.0 + kick * 0.5  # Extra boost for tour edges
    
    # Identify edges in the local-optimum tour
    tour_edges = set()
    for i in range(len(local_opt_tour)):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % len(local_opt_tour)]
        tour_edges.add((min(u, v), max(u, v)))
    
    # Apply extra penalty to tour edges
    for (u, v) in tour_edges:
        # Blend the base penalty with the tour-edge boost
        current_penalty = penalty_factor[u, v]
        boosted_penalty = max(current_penalty, tour_penalty_boost)
        modified[u, v] *= boosted_penalty
        modified[v, u] *= boosted_penalty
    
    # Ensure symmetry
    modified = (modified + modified.T) / 2.0
    
    # Ensure no negative or infinite values
    modified = np.maximum(modified, 1e-10)
    modified = np.nan_to_num(modified, nan=1.0, posinf=1e10, neginf=1e10)
    
    return modified
# EVOLVE-BLOCK-END
