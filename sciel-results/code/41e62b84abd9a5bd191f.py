"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Args:
        distance_matrix: A 2D array of distances between nodes.
        coordinates: A 2D array of coordinates for each node.
        demands: A 1D array of demands for each node.
        capacity: The capacity of the vehicles.
        
    Returns:
        A 2D array of the same shape as distance_matrix with heuristic values.
    """
    # Ensure inputs are numpy arrays for consistent operations
    d = np.asarray(distance_matrix, dtype=float)
    
    # Add a small epsilon to avoid division by zero
    # The problem states non-positive entries become 1e-9, but we should ensure
    # the heuristic calculation itself is robust.
    eps = 1e-9
    
    # Use a power law heuristic with alpha=2.
    # eta_ij = (1 / d_ij)^alpha
    # This makes shorter edges significantly more attractive than longer ones,
    # which helps in constructing shorter routes by favoring local moves.
    alpha = 2.0
    
    # Calculate the heuristic matrix
    # We use np.power for clarity and potential numerical stability
    heuristic = np.power(1.0 / (d + eps), alpha)
    
    # Ensure no negative or zero values remain (though 1/(d+eps) is always positive)
    # The problem states "Heuristic entries that are not positive become 1e-9"
    # Since d >= 0, 1/(d+eps) > 0.
    
    # Replace any non-finite values (like inf if d was 0 and eps was 0, but we added eps)
    # Just in case
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
