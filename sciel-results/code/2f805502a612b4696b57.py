"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement with an exploration bonus.
    
    Args:
        mu: Array of mean predictions (shape: (2000,))
        sigma: Array of standard deviation predictions (shape: (2000,))
        f_best: The lowest observed value so far.
        
    Returns:
        Array of acquisition scores (shape: (2000,)), higher is better.
    """
    # Avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: Since we are minimizing, improvement is f_best - mu.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration Bonus
    # Adding a small multiple of the standard deviation encourages exploration
    # of regions with high uncertainty, helping to escape local minima
    # in the multimodal RBF landscape.
    exploration_weight = 0.15
    score = ei + exploration_weight * sigma_safe
    
    # Ensure the output is finite and handles any numerical edge cases
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
