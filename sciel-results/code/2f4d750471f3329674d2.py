"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ACO orienteering.
    
    The heuristic favors edges leading to high-prize nodes at short distances.
    Uses a power-law distance penalty to emphasize efficient paths.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    eps = 1e-9
    
    # Distance penalty: shorter distances are more attractive
    # Use power law with exponent > 1 to emphasize short distances
    dist_safe = np.maximum(distance_matrix, eps)
    dist_penalty = np.power(dist_safe, 1.5)
    
    # Prize attractiveness: use prize of destination node
    # Add small epsilon to avoid zero division and ensure positivity
    prize_attr = (prizes[None, :] + eps)
    
    # Heuristic: prize / distance^alpha
    # This rewards high-prize nodes that are close
    heuristic = prize_attr / dist_penalty
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, eps)
    
    # Normalize to reasonable scale (optional, but helps with numerical stability)
    # The ACO uses pheromone * heuristic, so scale matters
    # Keep values in a reasonable range
    max_val = np.max(heuristic)
    if max_val > 0:
        heuristic = heuristic / max_val * 10.0  # Scale so max is ~10
    
    # Ensure minimum value
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic
# EVOLVE-BLOCK-END
