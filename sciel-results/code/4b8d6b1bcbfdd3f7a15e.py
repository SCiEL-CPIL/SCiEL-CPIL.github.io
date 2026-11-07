"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient (KG) inspired acquisition function with sublinear cost discount.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions for 2000 candidates.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions for 2000 candidates.
    f_best : float
        Current best observed value (lowest).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    # Ensure sigma is strictly positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement: how much better than current best, in std units
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # This is the 1-step Knowledge Gradient approximation
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware discount
    # We use a linear cost penalty (1/cost) to maximize improvement per unit cost.
    # This is more aggressive than 1/sqrt(cost) and prioritizes cheaper evaluations
    # when the expected improvement is similar.
    cost_safe = np.maximum(cost, 1e-9)
    cost_discount = 1.0 / cost_safe
    
    # Final score: EI discounted by cost
    scores = ei * cost_discount
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
