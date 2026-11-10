"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Computes the heuristic matrix for Ant Colony Optimization.
    
    Uses an inverse squared distance heuristic to aggressively suppress
    long-range edges, promoting local connectivity and non-crossing tours
    typical of optimal 2D TSP solutions.
    """
    # Add a small epsilon to avoid division by zero for any diagonal or coincident points
    # although TSP usually has distinct points, this ensures robustness.
    epsilon = 1e-9
    
    # Use power 2.0 for stronger locality bias compared to the baseline power 1.0.
    # This makes the heuristic decay faster with distance.
    power = 2.0
    
    # Calculate heuristic: 1 / (distance^power)
    # We clip the distance to epsilon to ensure positivity and finiteness
    safe_dist = np.maximum(distance_matrix, epsilon)
    eta = 1.0 / (safe_dist ** power)
    
    # Ensure the output is a valid numpy array with positive finite values
    # The problem statement says "Heuristic entries that are not positive become 1e-9"
    # Our calculation ensures positive values, but we enforce it just in case of numerical issues
    eta[~np.isfinite(eta) | (eta <= 0)] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
