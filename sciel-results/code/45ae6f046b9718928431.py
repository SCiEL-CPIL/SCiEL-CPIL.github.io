"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for TSP using a negative-curvature /
    high-power distance penalty to suppress long-range crossing edges.
    
    The heuristic is inversely proportional to distance raised to a power,
    which penalizes long edges. We use d^2 (squared distance) which corresponds
    to the squared Euclidean distance, a natural geometric quantity that
    balances local connectivity with global tour structure.
    
    Using d^2 instead of d^3 or higher allows the ants to explore slightly
    longer edges when necessary to complete a tour, while still strongly
    preferring short local connections. This should lead to better tours
    by avoiding the extreme bias toward nearest-neighbor chains that can
    trap the colony in suboptimal solutions.
    """
    # Ensure we're working with a numpy array
    d = np.asarray(distance_matrix, dtype=np.float64)
    
    # Guard against zero distances (shouldn't happen with random points, but just in case)
    d_safe = np.maximum(d, 1e-9)
    
    # Use squared distance (d^2) for the heuristic.
    # This is the natural "sum of squared edge lengths" mentioned in the directive.
    # It provides a strong penalty for long edges while being less extreme than
    # higher powers, allowing the ant colony to find better global tours.
    
    heuristic = 1.0 / (d_safe ** 2)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
