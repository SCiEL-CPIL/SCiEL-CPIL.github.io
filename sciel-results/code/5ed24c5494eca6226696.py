"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO.
    
    Uses a savings-based heuristic combined with inverse distance.
    The savings metric s(i, j) = d(0, i) + d(0, j) - d(i, j) indicates
    how beneficial it is to link i and j directly.
    
    We use a soft-max transformation to ensure positivity and smoothness.
    """
    n = distance_matrix.shape[0]
    depot = 0
    
    # Calculate savings for all pairs (i, j)
    # s(i, j) = d(i, 0) + d(0, j) - d(i, j)
    # Note: distance_matrix is symmetric, so d(i, 0) == d(0, i)
    d_to_depot = distance_matrix[:, depot]
    savings = d_to_depot[:, np.newaxis] + d_to_depot[np.newaxis, :] - distance_matrix
    
    # The savings can be negative if the triangle inequality is violated or due to noise,
    # but in Euclidean space it's always >= 0. However, for robustness, we clip.
    # A large savings means a strong preference to connect i and j.
    
    # We want the heuristic to be larger for more attractive edges.
    # Standard ACO heuristic is eta = 1/d.
    # Savings-based heuristic often uses exp(savings / scale) or just savings.
    
    # To combine distance preference (shorter is better) and savings (larger is better):
    # We can use: eta[i, j] = (1 + savings[i, j]) / (distance_matrix[i, j] + epsilon)
    # Or simply use the savings as the primary driver if we want to encourage route building.
    
    # Let's try a combination that is robust:
    # eta = (1.0 + savings) / (distance_matrix + 1e-9)
    # This favors short edges (denominator) but boosts them if they have high savings.
    
    epsilon = 1e-9
    # Ensure denominator is positive
    eta = (1.0 + savings) / (distance_matrix + epsilon)
    
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    # Since savings >= 0 and dist > 0, eta is always positive.
    # However, we must handle the depot row/col carefully.
    # In ACO, ants pick the next customer. Edges to/from depot are also used.
    # Savings for edges involving depot:
    # s(i, 0) = d(i, 0) + d(0, 0) - d(i, 0) = 0.
    # So eta[i, 0] = 1 / d(i, 0). This is standard inverse distance for returning to depot.
    # s(0, j) = 0. eta[0, j] = 1 / d(0, j).
    
    # This seems reasonable.
    
    # Normalize or scale?
    # The pheromone starts at 1.0.
    # If eta is very large, it might dominate. If very small, pheromone dominates.
    # 1/d is typically O(1). Savings is O(1). So eta is O(1).
    
    # Ensure no non-finite values
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    
    # Enforce positivity
    eta = np.where(eta > 0, eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
