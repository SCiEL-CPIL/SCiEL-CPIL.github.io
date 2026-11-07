"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for ant colony optimization
    of the orienteering problem.
    
    The heuristic is designed to:
    1. Strongly favor high-prize nodes (cubic exponent)
    2. Penalize distance with a super-linear exponent (1.5)
    3. Account for the round-trip constraint (d[i,j] + d[j,0] <= max_len)
    4. Ensure all entries are positive and finite
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Prize term: cube of prize to strongly emphasize high-value nodes
    # Add small epsilon to avoid zero (node 0 has prize 0)
    prize_term = (prizes + 1e-6) ** 3
    
    # Distance term: use the effective round-trip cost
    # For moving from i to j, the additional cost is d[i,j] + d[j,0]
    # This captures the feasibility constraint: length_so_far + d[i,j] + d[j,0] <= max_len
    # Since we don't know length_so_far, we use a static proxy
    round_trip_cost = distance_matrix + distance_matrix[:, 0][None, :]
    
    # Penalize distance with exponent 1.5 to create strong gradient toward nearby nodes
    # Add epsilon to avoid division by zero
    dist_term = (round_trip_cost + 1e-4) ** 1.5
    
    # Compute heuristic: prize / distance
    heuristic = prize_term[None, :] / dist_term
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
