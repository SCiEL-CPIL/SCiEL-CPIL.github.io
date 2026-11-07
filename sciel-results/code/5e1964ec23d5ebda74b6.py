"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Compute a heuristic edge-attractiveness matrix for ACO on TSP.
    Returns a matrix of the same shape where larger values indicate more attractive edges.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Start with strong inverse distance bias
    # Using inverse cubed gives strong preference for short edges
    inv_d = 1.0 / (d + 1e-9)
    
    # Normalize so that the max is 1.0
    max_val = np.max(inv_d)
    if max_val > 0:
        inv_d = inv_d / max_val
    
    # Add a small component based on the overall structure
    # This helps break symmetry and provides a gentle global bias
    # Compute a smoothed version using a Gaussian-like kernel on distances
    # d_max is the maximum distance in the matrix
    d_max = np.max(d)
    if d_max > 0:
        # Scale distances to [0, 1]
        d_scaled = d / d_max
        # Create a smooth attraction: exp(-alpha * d_scaled)
        alpha = 2.0
        smooth_attr = np.exp(-alpha * d_scaled)
        
        # Combine: primary is inverse distance, secondary is smooth exponential
        # The smooth component helps with longer-range structure
        combined = 0.8 * inv_d + 0.2 * smooth_attr
    else:
        combined = inv_d
    
    # Ensure all values are positive
    combined = np.maximum(combined, 1e-9)
    
    return combined
# EVOLVE-BLOCK-END
