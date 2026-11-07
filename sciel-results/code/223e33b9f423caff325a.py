"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a
    local exploration bonus based on predictive variance.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (N,)
        Mean of the GP posterior at N candidate points.
    sigma : np.ndarray, shape (N,)
        Standard deviation of the GP posterior at N candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (N,)
        Acquisition values for the N candidate points. Higher is better.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle numerical issues: ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Local exploration bonus: encourage points with high uncertainty
    # but only if they're not too far from the best observed value
    # This helps escape local optima in early stages
    # The bonus is proportional to sigma but damped by how far mu is from f_best
    # We use a soft cap to prevent sigma from dominating entirely
    
    # Compute a relative uncertainty measure
    # Scale sigma by the range of mu to make it dimensionless-ish
    mu_range = mu.max() - mu.min() if mu.size > 0 else 1.0
    if mu_range < 1e-12:
        mu_range = 1.0
    
    # Exploration bonus: high when sigma is large relative to the scale
    # But we don't want to over-explore if mu is already much better than f_best
    # Use a Gaussian-like damping: bonus peaks when mu is near f_best
    # and decays as mu moves away from f_best
    
    # Distance from best in standardized units
    dist_from_best = np.abs(mu - f_best) / (mu_range + 1e-12)
    
    # Exploration weight: higher when closer to best and higher uncertainty
    # Use a simple product: sigma * exp(-dist_from_best)
    # This gives high scores to uncertain points near the current best
    exploration_bonus = sigma_safe * np.exp(-dist_from_best)
    
    # Combine EI and exploration bonus
    # Weight the exploration term to be moderate
    # Early on, we want more exploration; later, more exploitation
    # But we don't have access to the step number, so use a fixed weight
    # The EI term naturally grows as f_best improves, so it will dominate later
    
    # Use a tunable weight for exploration
    # A weight of 0.5 means exploration is half as important as EI at the start
    # This can be adjusted based on performance
    exploration_weight = 0.5
    
    scores = ei + exploration_weight * exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
