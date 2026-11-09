"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Pareto-frontier guided exploration strategy.
    
    1. Identify the candidate with the lowest predicted mean (mu_min) as an aggressive exploitation anchor.
    2. Compute the acquisition score as the product of sigma and a Gaussian decay of the distance from this anchor.
    
    This geometric approach prioritizes high-uncertainty regions that are close to the best predicted mean,
    effectively finding the upper confidence bound of the minimum.
    """
    # Handle edge cases where sigma might be zero or very small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Step 1: Find the anchor - candidate with lowest predicted mean
    mu_min = np.min(mu)
    
    # Step 2: Compute the standardized distance from the anchor
    # z = (mu - mu_min) / sigma
    # Points with mu close to mu_min have z close to 0.
    # Points with mu >> mu_min have large positive z.
    z = (mu - mu_min) / sigma_safe
    
    # Step 3: Compute the Gaussian weight (decay of distance)
    # exp(-0.5 * z^2) is 1 when z=0 and decays rapidly as z increases.
    # This ensures we only strongly explore points that are *potentially* better than the current best prediction
    # (i.e., close to the frontier).
    gaussian_weight = np.exp(-0.5 * z**2)
    
    # Step 4: Compute acquisition score as product of sigma and the Gaussian weight
    # This rewards points that have high uncertainty AND are close to the best predicted mean.
    score = sigma_safe * gaussian_weight
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
