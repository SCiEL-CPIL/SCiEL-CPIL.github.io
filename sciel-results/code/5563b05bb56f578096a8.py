"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Improved Pareto-frontier guided exploration strategy.
    
    1. Identify the best reference point: the minimum of the predicted mean (mu_min) and the best observed value (f_best).
       This ensures the search is anchored to the best known or predicted performance.
    2. Compute the acquisition score as the product of sigma and a Gaussian decay of the distance from this reference.
    
    This geometric approach prioritizes high-uncertainty regions that are close to the best known/performance frontier.
    """
    # Handle edge cases where sigma might be zero or very small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Step 1: Find the reference anchor
    # Use the best predicted mean, but don't go lower than the best observed value?
    # Actually, if mu_min < f_best, we trust the prediction (exploitation of GP).
    # If mu_min > f_best, we trust the observation (exploitation of data).
    # This creates a "hard floor" at f_best for the anchor if the GP is pessimistic.
    mu_min = np.min(mu)
    anchor = min(mu_min, f_best)
    
    # Step 2: Compute the standardized distance from the anchor
    z = (mu - anchor) / sigma_safe
    
    # Step 3: Compute the Gaussian weight (decay of distance)
    gaussian_weight = np.exp(-0.5 * z**2)
    
    # Step 4: Compute acquisition score as product of sigma and the Gaussian weight
    score = sigma_safe * gaussian_weight
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
