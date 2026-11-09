"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function with dynamic exploration constant.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far (lowest since we minimize).
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        UCB scores; higher values indicate more promising candidates.
    """
    # Dynamic UCB with exploration constant decreasing as uncertainty drops.
    # We use the mean posterior standard deviation as a proxy for the
    # "age" of the optimization. Early on, sigma is high (near 1 for
    # standardized targets), so kappa is high (exploration). Later, sigma
    # drops, so kappa drops (exploitation).
    # This approximates a kappa ~ 1/sqrt(t) schedule, which is standard
    # for balancing exploration and exploitation in BO.
    
    sigma_mean = np.mean(sigma)
    
    # Ensure sigma_mean is not zero to avoid issues, though it shouldn't be.
    if sigma_mean < 1e-6:
        sigma_mean = 1e-6
        
    # Tune kappa:
    # Start with ~4.0 when sigma_mean ~ 1.0
    # End with ~1.0 when sigma_mean ~ 0.0
    # Linear interpolation: kappa = 3.0 * sigma_mean + 1.0
    kappa = 3.0 * sigma_mean + 1.0
    
    # Compute UCB scores for minimization: mu - kappa * sigma
    scores = mu - kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
