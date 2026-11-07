"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, weights):
    """
    Computes one attractiveness score per item for the ant colony optimization.
    
    Args:
        prizes: np.array of shape (5, 100) with prizes in [0, 1].
        weights: np.array of shape (5, 100) with weights in [0, 1].
        
    Returns:
        np.array of shape (100,) with heuristic scores.
    """
    # Calculate total prize per item
    total_prize = np.sum(prizes, axis=0)
    
    # Calculate total weight per item across all constraints
    total_weight = np.sum(weights, axis=0)
    
    # Heuristic: Stabilized Prize-to-Weight Ratio.
    # Adding 1.0 to the denominator stabilizes the ratio for items with very low weight,
    # preventing noise-driven selection of negligible-weight items and providing a smoother
    # gradient for the ACO pheromone update.
    eta = total_prize / (total_weight + 1.0)
    
    # Ensure all values are positive as required by the contract
    eta = np.maximum(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
