"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for TSP using a negative-curvature /
    high-power distance penalty to suppress long-range crossing edges.
    
    The heuristic is inversely proportional to distance raised to a high power,
    which aggressively penalizes long edges. This leverages the fact that optimal
    TSP tours in 2D are planar and prefer short, local edges.
    
    We use d^4 (or similar high power) to strongly suppress non-local connections.
    """
    # Ensure we're working with a numpy array
    d = np.asarray(distance_matrix, dtype=np.float64)
    
    # Guard against zero distances (shouldn't happen with random points, but just in case)
    d_safe = np.maximum(d, 1e-9)
    
    # Use a high power of distance to aggressively suppress long-range edges.
    # We increase the power from 3 to 4 to further suppress non-local connections.
    # This enforces a stronger local preference, which can help avoid crossing edges
    # in the 2D TSP instances.
    
    heuristic = 1.0 / (d_safe ** 4)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
