"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using a Gaussian Kernel Distance Heuristic.
    
    This approach uses a Gaussian kernel to weight edges based on distance.
    Unlike inverse distance, this provides a smooth decay that emphasizes
    local proximity without the extreme values of 1/d. It does not rely
    on clustering, offering a distinct geometric bias.
    """
    n = distance_matrix.shape[0]
    
    # Create a copy to avoid modifying input
    d = distance_matrix.copy()
    
    # Set diagonal to a large value to ignore self-loops for statistics
    np.fill_diagonal(d, np.inf)
    
    # Calculate a scale parameter sigma based on the median distance
    # This makes the heuristic adaptive to the scale of the instance
    finite_dists = d[np.isfinite(d)]
    if len(finite_dists) > 0:
        sigma = np.median(finite_dists)
    else:
        sigma = 1.0
        
    # Avoid division by zero if sigma is 0
    if sigma < 1e-9:
        sigma = 1.0
        
    # Compute Gaussian kernel: exp(-d^2 / (2 * sigma^2))
    # We use the provided distance_matrix directly for the final calculation
    # to ensure consistency with the ant's path cost
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        heuristic = np.exp(-(distance_matrix ** 2) / (2.0 * sigma ** 2))
        
    # Handle any non-finite values (e.g., from inf distances)
    heuristic[~np.isfinite(heuristic)] = 1e-9
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(heuristic, 0.0)
    
    return heuristic
# EVOLVE-BLOCK-END
