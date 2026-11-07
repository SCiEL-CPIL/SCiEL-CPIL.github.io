import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores for Bayesian optimization of hartmann 6d.
    Combines Expected Improvement (EI) with an adaptive exploration term.
    
    Parameters:
    mu: np.ndarray of shape (N,) - Posterior mean
    sigma: np.ndarray of shape (N,) - Posterior standard deviation
    f_best: float - Best observed value so far (minimum)
    
    Returns:
    np.ndarray of shape (N,) - Acquisition scores (higher is better)
    """
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # 2. Adaptive Exploration Bonus
    # Add a small bonus for high uncertainty, scaled by the relative uncertainty.
    # This helps avoid getting stuck in local optima early on, but decays as confidence grows.
    
    # Calculate mean sigma to gauge overall uncertainty
    sigma_mean = np.mean(sigma_safe)
    
    # Calculate a measure of the scale of the improvement (std of mu + sigma)
    # This normalizes the exploration bonus so it doesn't dominate if the objective scale changes
    scale = np.std(mu) + sigma_mean + eps
    
    if sigma_mean > 0:
        # Exploration weight starts around 0.1 and scales with relative uncertainty
        # As sigma_mean becomes small relative to the scale, exploration decreases
        exploration_weight = 0.1 * (sigma_mean / scale)
    else:
        exploration_weight = 0.0
    
    exploration_bonus = exploration_weight * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    # Replace any NaN or Inf with a very low score so they are not selected
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
