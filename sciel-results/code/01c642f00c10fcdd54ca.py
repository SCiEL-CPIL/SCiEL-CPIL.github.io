"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    Uses a fixed exploration constant k=3.0 to balance exploration and exploitation
    for the multimodal Michalewicz function.
    
    Args:
        mu: Mean of the Gaussian Process posterior (shape: 2000,).
        sigma: Standard deviation of the GP posterior (shape: 2000,).
        f_best: Best (lowest) function value observed so far.
        
    Returns:
        scores: Array of 2000 scores. Higher score indicates a better candidate.
    """
    # Ensure sigma is non-negative to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Fixed exploration constant for LCB.
    # k=3.0 is chosen to aggressively target high-uncertainty regions
    # while still considering the posterior mean.
    k = 3.0
    
    # LCB for minimization: Q(x) = mu(x) - k * sigma(x)
    # We want to select the point with the *lowest* Q(x).
    # The acquisition score to be *maximized* is therefore -Q(x) = -mu(x) + k * sigma(x).
    scores = -mu + k * sigma
    
    # Ensure all scores are finite to satisfy the contract.
    # Replace any non-finite values (inf, nan) with a very low score.
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
