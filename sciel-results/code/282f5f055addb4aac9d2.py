"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Scaled Thompson Sampling acquisition function.
    
    We draw a single sample from the posterior at each candidate point,
    but scale the uncertainty term to reduce the variance of the acquisition
    function. This provides a better balance between exploration and
    exploitation for small evaluation budgets by preventing the algorithm
    from being overly dominated by stochastic noise in high-uncertainty regions.
    
    The score is based on:
        theta = mu + kappa * sigma * Z, where Z ~ N(0, 1) and kappa < 1
    
    Parameters:
        mu: array of shape (2000,) - posterior mean in objective units
        sigma: array of shape (2000,) - posterior std in objective units
        f_best: float - lowest observed value so far
    
    Returns:
        scores: array of shape (2000,) - finite scores, higher = evaluate next
    """
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Exploration scaling factor: 0.5 reduces the effective variance of the sample
    # by a factor of 4, making the selection more deterministic (exploitation-heavy)
    # while retaining some stochasticity.
    kappa = 0.5
    
    # Draw standard normal noise
    Z = np.random.standard_normal(len(mu))
    
    # Scaled Thompson sample
    theta = mu + kappa * sigma_safe * Z
    
    # Score: negative of the sample (since we minimize, lower theta is better)
    scores = -theta
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
